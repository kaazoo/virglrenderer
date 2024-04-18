/*
 * Copyright 2024 Sergio Lopez
 * Copyright 2022 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <linux/dma-buf.h>
#include <sys/mman.h>
#include <sys/types.h>

#include <xf86drm.h>

#include "virgl_context.h"
#include "virgl_util.h"
#include "virglrenderer.h"

#include "util/anon_file.h"
#include "util/hash_table.h"
#include "util/u_atomic.h"

#include "drm_fence.h"

#include "asahi_drm.h"
#include "asahi_proto.h"
#include "asahi_renderer.h"

static unsigned nr_timelines;

/**
 * A single context (from the PoV of the virtio-gpu protocol) maps to
 * a single drm device open.  Other drm constructs (ie. submitqueue) are
 * opaque to the protocol.
 *
 * Typically each guest process will open a single virtio-gpu "context".
 * The single drm device open maps to an individual GEM address_space on
 * the kernel side, providing GPU address space isolation between guest
 * processes.
 *
 * GEM buffer objects are tracked via one of two id's:
 *  - resource-id:  global, assigned by guest kernel
 *  - blob-id:      context specific, assigned by guest userspace
 *
 * The blob-id is used to link the bo created via the corresponding ioctl
 * and the get_blob() cb. It is unused in the case of a bo that is
 * imported from another context.  An object is added to the blob table
 * in GEM_NEW and removed in ctx->get_blob() (where it is added to
 * resource_table). By avoiding having an obj in both tables, we can
 * safely free remaining entries in either hashtable at context teardown.
 */
struct asahi_context {
   struct virgl_context base;

   struct asahi_shmem *shmem;
   uint8_t *rsp_mem;
   uint32_t rsp_mem_sz;

   struct vdrm_ccmd_rsp *current_rsp;

   int fd;

   struct hash_table *blob_table;
   struct hash_table *resource_table;

   int eventfd;

   /**
    * Indexed by ring_idx-1, which is the same as the submitqueue priority+1.
    * On the kernel side, there is some drm_sched_entity per {drm_file, prio}
    * tuple, and the sched entity determines the fence timeline, ie. submits
    * against a single sched entity complete in fifo order.
    */
   struct drm_timeline timelines[];
};
DEFINE_CAST(virgl_context, asahi_context)

static struct hash_entry *
table_search(struct hash_table *ht, uint32_t key)
{
   /* zero is not a valid key for u32_keys hashtable: */
   if (!key)
      return NULL;
   return _mesa_hash_table_search(ht, (void *)(uintptr_t)key);
}

static int
gem_close(int fd, uint32_t handle)
{
   struct drm_gem_close args = { .handle = handle };
   return drmIoctl(fd, DRM_IOCTL_GEM_CLOSE, &args);
}

struct asahi_object {
   uint32_t blob_id;
   uint32_t res_id;
   uint32_t handle;
   uint32_t flags;
   uint32_t size;
   bool exported   : 1;
   bool exportable : 1;
   uint8_t *map;
};

static struct asahi_object *
asahi_object_create(uint32_t handle, uint32_t flags, uint32_t size)
{
   struct asahi_object *obj = calloc(1, sizeof(*obj));

   if (!obj)
      return NULL;

   obj->handle = handle;
   obj->flags = flags;
   obj->size = size;

   return obj;
}

static bool
valid_blob_id(struct asahi_context *actx, uint32_t blob_id)
{
   /* must be non-zero: */
   if (blob_id == 0)
      return false;

   /* must not already be in-use: */
   if (table_search(actx->blob_table, blob_id))
      return false;

   return true;
}

static void
asahi_object_set_blob_id(struct asahi_context *actx, struct asahi_object *obj,
                         uint32_t blob_id)
{
   assert(valid_blob_id(actx, blob_id));

   obj->blob_id = blob_id;
   _mesa_hash_table_insert(actx->blob_table, (void *)(uintptr_t)obj->blob_id, obj);
}

static bool
valid_res_id(struct asahi_context *actx, uint32_t res_id)
{
   return !table_search(actx->resource_table, res_id);
}

static void
asahi_object_set_res_id(struct asahi_context *actx, struct asahi_object *obj,
                        uint32_t res_id)
{
   assert(valid_res_id(actx, res_id));

   obj->res_id = res_id;
   _mesa_hash_table_insert(actx->resource_table, (void *)(uintptr_t)obj->res_id, obj);
}

static void
asahi_remove_object(struct asahi_context *actx, struct asahi_object *obj)
{
   drm_dbg("obj=%p, blob_id=%u, res_id=%u", (void *)obj, obj->blob_id, obj->res_id);
   _mesa_hash_table_remove_key(actx->resource_table, (void *)(uintptr_t)obj->res_id);
}

static struct asahi_object *
asahi_retrieve_object_from_blob_id(struct asahi_context *actx, uint64_t blob_id)
{
   assert((blob_id >> 32) == 0);
   uint32_t id = blob_id;
   struct hash_entry *entry = table_search(actx->blob_table, id);
   if (!entry)
      return NULL;
   struct asahi_object *obj = entry->data;
   _mesa_hash_table_remove(actx->blob_table, entry);
   return obj;
}

static struct asahi_object *
asahi_get_object_from_res_id(struct asahi_context *actx, uint32_t res_id)
{
   const struct hash_entry *entry = table_search(actx->resource_table, res_id);
   return likely(entry) ? entry->data : NULL;
}

static uint32_t
handle_from_res_id(struct asahi_context *actx, uint32_t res_id)
{
   struct asahi_object *obj = asahi_get_object_from_res_id(actx, res_id);
   if (!obj)
      return 0; /* zero is an invalid GEM handle */
   return obj->handle;
}

/**
 * Probe capset params.
 */
int
asahi_renderer_probe(UNUSED int fd, struct virgl_renderer_capset_drm *capset)
{
   drm_log("");

   capset->wire_format_version = 2;

   nr_timelines = 1;

   return 0;
}

static void
asahi_renderer_unmap_blob(struct asahi_context *actx)
{
   if (!actx->shmem)
      return;

   uint32_t blob_size = actx->rsp_mem_sz + actx->shmem->base.rsp_mem_offset;

   munmap(actx->shmem, blob_size);

   actx->shmem = NULL;
   actx->rsp_mem = NULL;
   actx->rsp_mem_sz = 0;
}

static void
resource_delete_fxn(struct hash_entry *entry)
{
   free((void *)entry->data);
}

static void
asahi_renderer_destroy(struct virgl_context *vctx)
{
   struct asahi_context *actx = to_asahi_context(vctx);

   for (unsigned i = 0; i < nr_timelines; i++)
      drm_timeline_fini(&actx->timelines[i]);

   close(actx->eventfd);

   asahi_renderer_unmap_blob(actx);

   _mesa_hash_table_destroy(actx->resource_table, resource_delete_fxn);
   _mesa_hash_table_destroy(actx->blob_table, resource_delete_fxn);

   close(actx->fd);
   free(actx);
}

static void
asahi_renderer_attach_resource(struct virgl_context *vctx, struct virgl_resource *res)
{
   struct asahi_context *actx = to_asahi_context(vctx);
   struct asahi_object *obj = asahi_get_object_from_res_id(actx, res->res_id);

   drm_dbg("obj=%p, res_id=%u", (void *)obj, res->res_id);

   if (!obj) {
      int fd;
      enum virgl_resource_fd_type fd_type = virgl_resource_export_fd(res, &fd);

      /* If importing a dmabuf resource created by another context (or
       * externally), then import it to create a gem obj handle in our
       * context:
       */
      if (fd_type == VIRGL_RESOURCE_FD_DMABUF) {
         uint32_t handle;
         int ret;

         ret = drmPrimeFDToHandle(actx->fd, fd, &handle);
         if (ret) {
            drm_log("Could not import: %s", strerror(errno));
            close(fd);
            return;
         }

         /* lseek() to get bo size */
         int size = lseek(fd, 0, SEEK_END);
         if (size < 0)
            drm_log("lseek failed: %d (%s)", size, strerror(errno));
         close(fd);

         obj = asahi_object_create(handle, 0, size);
         if (!obj)
            return;

         asahi_object_set_res_id(actx, obj, res->res_id);

         drm_dbg("obj=%p, res_id=%u, handle=%u", (void *)obj, obj->res_id, obj->handle);
      } else {
         if (fd_type != VIRGL_RESOURCE_FD_INVALID)
            close(fd);
         return;
      }
   }
}

static void
asahi_renderer_detach_resource(struct virgl_context *vctx, struct virgl_resource *res)
{
   struct asahi_context *actx = to_asahi_context(vctx);
   struct asahi_object *obj = asahi_get_object_from_res_id(actx, res->res_id);

   drm_dbg("obj=%p, res_id=%u", (void *)obj, res->res_id);

   if (!obj)
      return;

   if (res->fd_type == VIRGL_RESOURCE_FD_SHM) {
      asahi_renderer_unmap_blob(actx);

      /* shmem resources don't have an backing host GEM bo:, so bail now: */
      return;
   }

   asahi_remove_object(actx, obj);

   if (obj->map)
      munmap(obj->map, obj->size);

   gem_close(actx->fd, obj->handle);

   free(obj);
}

static void *
asahi_renderer_resource_map(struct virgl_context *vctx, struct virgl_resource *res,
                            void *addr, int32_t prot, int32_t flags)
{
   struct asahi_context *actx = to_asahi_context(vctx);
   struct asahi_object *obj = asahi_get_object_from_res_id(actx, res->res_id);

   drm_dbg("obj=%p, res_id=%u", (void *)obj, res->res_id);

   if (!obj) {
      drm_log("invalid res_id %u", res->res_id);
      return NULL;
   }

   struct drm_asahi_gem_mmap_offset gem_offset = { .handle = obj->handle };
   int ret = drmIoctl(actx->fd, DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET, &gem_offset);
   if (ret) {
      drm_log("Failed to obtain GEM offset: %s", strerror(errno));
      return NULL;
   }

   void *map = mmap(addr, res->map_size, prot, flags, actx->fd, gem_offset.offset);
   if (map == MAP_FAILED) {
      drm_log("Failed to mmap GEM object: %s", strerror(errno));
      return NULL;
   }

   return map;
}

static enum virgl_resource_fd_type
asahi_renderer_export_opaque_handle(struct virgl_context *vctx,
                                    struct virgl_resource *res, int *out_fd)
{
   struct asahi_context *actx = to_asahi_context(vctx);
   struct asahi_object *obj = asahi_get_object_from_res_id(actx, res->res_id);
   int ret;

   drm_dbg("obj=%p, res_id=%u", (void *)obj, res->res_id);

   if (!obj) {
      drm_log("invalid res_id %u", res->res_id);
      return VIRGL_RESOURCE_FD_INVALID;
   }

   if (!obj->exportable) {
      *out_fd = -1;
      return VIRGL_RESOURCE_OPAQUE_HANDLE;
   }

   ret = drmPrimeHandleToFD(actx->fd, obj->handle, DRM_CLOEXEC | DRM_RDWR, out_fd);
   if (ret) {
      drm_log("failed to get dmabuf fd: %s", strerror(errno));
      return VIRGL_RESOURCE_FD_INVALID;
   }

   return VIRGL_RESOURCE_FD_DMABUF;
}

static int
asahi_renderer_transfer_3d(UNUSED struct virgl_context *vctx,
                           UNUSED struct virgl_resource *res,
                           UNUSED const struct vrend_transfer_info *info,
                           UNUSED int transfer_mode)
{
   drm_log("unsupported");
   return -1;
}

static int
asahi_renderer_get_blob(struct virgl_context *vctx, uint32_t res_id, uint64_t blob_id,
                        uint64_t blob_size, uint32_t blob_flags,
                        struct virgl_context_blob *blob)
{
   struct asahi_context *actx = to_asahi_context(vctx);

   drm_dbg("blob_id=%" PRIu64 ", res_id=%u, blob_size=%" PRIu64 ", blob_flags=0x%x",
           blob_id, res_id, blob_size, blob_flags);

   if ((blob_id >> 32) != 0) {
      drm_log("invalid blob_id: %" PRIu64, blob_id);
      return -EINVAL;
   }

   /* blob_id of zero is reserved for the shmem buffer: */
   if (blob_id == 0) {
      int fd;

      if (blob_flags != VIRGL_RENDERER_BLOB_FLAG_USE_MAPPABLE) {
         drm_log("invalid blob_flags: 0x%x", blob_flags);
         return -EINVAL;
      }

      if (actx->shmem) {
         drm_log("There can be only one!");
         return -EINVAL;
      }

      if (blob_size < sizeof(*actx->shmem)) {
         drm_log("Invalid blob size");
         return -EINVAL;
      }

      fd = os_create_anonymous_file(blob_size, "asahi-shmem");
      if (fd < 0) {
         drm_log("Failed to create shmem file: %s", strerror(errno));
         return -ENOMEM;
      }

      int ret = fcntl(fd, F_ADD_SEALS, F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW);
      if (ret) {
         drm_log("fcntl failed: %s", strerror(errno));
         close(fd);
         return -ENOMEM;
      }

      actx->shmem = mmap(NULL, blob_size, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
      if (actx->shmem == MAP_FAILED) {
         drm_log("shmem mmap failed: %s", strerror(errno));
         close(fd);
         return -ENOMEM;
      }

      actx->shmem->base.rsp_mem_offset = sizeof(*actx->shmem);

      uint8_t *ptr = (uint8_t *)actx->shmem;
      actx->rsp_mem = &ptr[actx->shmem->base.rsp_mem_offset];
      actx->rsp_mem_sz = blob_size - actx->shmem->base.rsp_mem_offset;

      blob->type = VIRGL_RESOURCE_FD_SHM;
      blob->u.fd = fd;
      blob->map_info = VIRGL_RENDERER_MAP_CACHE_CACHED;

      return 0;
   }

   if (!valid_res_id(actx, res_id)) {
      drm_log("Invalid res_id %u", res_id);
      return -EINVAL;
   }

   struct asahi_object *obj = asahi_retrieve_object_from_blob_id(actx, blob_id);

   /* If GEM_NEW fails, we can end up here without a backing obj: */
   if (!obj) {
      drm_log("No object");
      return -ENOENT;
   }

   /* a memory can only be exported once; we don't want two resources to point
    * to the same storage.
    */
   if (obj->exported) {
      drm_log("Already exported!");
      return -EINVAL;
   }

   /* The size we get from guest userspace is not necessarily rounded up to the
    * nearest page size, but the actual GEM buffer allocation is, as is the
    * guest GEM buffer (and therefore the blob_size value we get from the guest
    * kernel).
    */
   if (ALIGN_POT(obj->size, getpagesize()) != blob_size) {
      drm_log("Invalid blob size");
      return -EINVAL;
   }

   asahi_object_set_res_id(actx, obj, res_id);

   if (blob_flags & VIRGL_RENDERER_BLOB_FLAG_USE_SHAREABLE) {
      int fd, ret;

      ret = drmPrimeHandleToFD(actx->fd, obj->handle, DRM_CLOEXEC | DRM_RDWR, &fd);
      if (ret) {
         drm_log("Export to fd failed");
         return -EINVAL;
      }

      blob->type = VIRGL_RESOURCE_FD_DMABUF;
      blob->u.fd = fd;
   } else {
      blob->type = VIRGL_RESOURCE_OPAQUE_HANDLE;
      blob->u.opaque_handle = obj->handle;
   }

   blob->map_info = VIRGL_RENDERER_MAP_CACHE_WC;

   obj->exported = true;
   obj->exportable = !!(blob_flags & VIRGL_RENDERER_BLOB_FLAG_USE_SHAREABLE);

   return 0;
}

static void *
asahi_context_rsp_noshadow(struct asahi_context *actx, const struct vdrm_ccmd_req *hdr)
{
   return &actx->rsp_mem[hdr->rsp_off];
}

static void *
asahi_context_rsp(struct asahi_context *actx, const struct vdrm_ccmd_req *hdr,
                  unsigned len)
{
   unsigned rsp_mem_sz = actx->rsp_mem_sz;
   unsigned off = hdr->rsp_off;

   if ((off > rsp_mem_sz) || (len > rsp_mem_sz - off)) {
      drm_log("invalid shm offset: off=%u, len=%u (shmem_size=%u)", off, len, rsp_mem_sz);
      return NULL;
   }

   struct vdrm_ccmd_rsp *rsp = asahi_context_rsp_noshadow(actx, hdr);

   assert(len >= sizeof(*rsp));

   /* With newer host and older guest, we could end up wanting a larger rsp struct
    * than guest expects, so allocate a shadow buffer in this case rather than
    * having to deal with this in all the different ccmd handlers.  This is similar
    * in a way to what drm_ioctl() does.
    */
   if (len > rsp->len) {
      rsp = malloc(len);
      if (!rsp)
         return NULL;
      rsp->len = len;
   }

   actx->current_rsp = rsp;

   return rsp;
}

static int
asahi_ccmd_nop(UNUSED struct asahi_context *actx, UNUSED const struct vdrm_ccmd_req *hdr)
{
   return 0;
}

static int
asahi_ccmd_ioctl_simple(struct asahi_context *actx, const struct vdrm_ccmd_req *hdr)
{
   const struct asahi_ccmd_ioctl_simple_req *req = to_asahi_ccmd_ioctl_simple_req(hdr);
   unsigned payload_len = _IOC_SIZE(req->cmd);
   unsigned req_len = size_add(sizeof(*req), payload_len);

   if (hdr->len != req_len) {
      drm_log("%u != %u", hdr->len, req_len);
      return -EINVAL;
   }

   /* Apply a reasonable upper bound on ioctl size: */
   if (payload_len > 128) {
      drm_log("invalid ioctl payload length: %u", payload_len);
      return -EINVAL;
   }

   /* Allow-list of supported ioctls: */
   unsigned iocnr = _IOC_NR(req->cmd) - DRM_COMMAND_BASE;
   switch (iocnr) {
   case DRM_ASAHI_GET_PARAMS:
   case DRM_ASAHI_VM_CREATE:
   case DRM_ASAHI_VM_DESTROY:
   case DRM_ASAHI_QUEUE_CREATE:
   case DRM_ASAHI_QUEUE_DESTROY:
      break;
   default:
      drm_log("invalid ioctl: %08x (%u)", req->cmd, iocnr);
      return -EINVAL;
   }

   struct asahi_ccmd_ioctl_simple_rsp *rsp;
   unsigned rsp_len = sizeof(*rsp);

   if (req->cmd & IOC_OUT)
      rsp_len = size_add(rsp_len, payload_len);

   rsp = asahi_context_rsp(actx, hdr, rsp_len);

   if (!rsp)
      return -ENOMEM;

   /* Copy the payload because the kernel can write (if IOC_OUT bit
    * is set) and to avoid casting away the const:
    */
   char payload[payload_len];
   memcpy(payload, req->payload, payload_len);

   rsp->ret = drmIoctl(actx->fd, req->cmd, payload);

   if (req->cmd & IOC_OUT)
      memcpy(rsp->payload, payload, payload_len);

   return 0;
}

static int
asahi_ccmd_get_params(struct asahi_context *actx, const struct vdrm_ccmd_req *hdr)
{
   struct asahi_ccmd_get_params_req *req = to_asahi_ccmd_get_params_req(hdr);
   unsigned req_len = sizeof(*req);

   if (hdr->len != req_len) {
      drm_log("%u != %u", hdr->len, req_len);
      return -EINVAL;
   }

   struct asahi_ccmd_get_params_rsp *rsp;
   unsigned rsp_len = sizeof(*rsp);

   rsp = asahi_context_rsp(actx, hdr, rsp_len);

   if (!rsp)
      return -ENOMEM;

   req->params.pointer = (uint64_t)&rsp->params;

   rsp->ret = drmIoctl(actx->fd, DRM_IOCTL_ASAHI_GET_PARAMS, &req->params);

   return 0;
}

static int
asahi_ccmd_gem_new(struct asahi_context *actx, const struct vdrm_ccmd_req *hdr)
{
   const struct asahi_ccmd_gem_new_req *req = to_asahi_ccmd_gem_new_req(hdr);
   int ret = 0;

   if (!valid_blob_id(actx, req->blob_id)) {
      drm_log("Invalid blob_id %u\n", req->blob_id);
      ret = -EINVAL;
      goto out_error;
   }

   int create_vm_id = req->vm_id;
   if (!(req->flags & ASAHI_GEM_VM_PRIVATE)) {
      create_vm_id = 0;
   }

   /*
    * First part, allocate the GEM bo:
    */
   struct drm_asahi_gem_create gem_create = {
      .flags = req->flags,
      .vm_id = create_vm_id,
      .size = req->size,
   };

   ret = drmIoctl(actx->fd, DRM_IOCTL_ASAHI_GEM_CREATE, &gem_create);
   if (ret) {
      drm_log("GEM_CREATE failed: %d (%s)\n", ret, strerror(errno));
      goto out_error;
   }

   /*
    * Second part, bind:
    */

   struct drm_asahi_gem_bind gem_bind = {
      .op = ASAHI_BIND_OP_BIND,
      .flags = req->bind_flags,
      .handle = gem_create.handle,
      .vm_id = req->vm_id,
      .offset = 0,
      .range = req->size,
      .addr = req->addr,
   };

   ret = drmIoctl(actx->fd, DRM_IOCTL_ASAHI_GEM_BIND, &gem_bind);
   if (ret) {
      drm_log("DRM_IOCTL_ASAHI_GEM_BIND failed: (handle=%d)\n", gem_create.handle);
      goto out_close;
   }

   /*
    * And then finally create our asahi_object for tracking the resource,
    * and add to blob table:
    */
   struct asahi_object *obj =
      asahi_object_create(gem_create.handle, req->flags, req->size);

   if (!obj) {
      ret = -ENOMEM;
      goto out_close;
   }

   asahi_object_set_blob_id(actx, obj, req->blob_id);

   drm_dbg("obj=%p, blob_id=%u, handle=%u\n", (void *)obj, obj->blob_id, obj->handle);

   return 0;

out_close:
   gem_close(actx->fd, gem_create.handle);
out_error:
   if (actx->shmem)
      actx->shmem->async_error++;
   return ret;
}

static int
asahi_ccmd_gem_bind(struct asahi_context *actx, const struct vdrm_ccmd_req *hdr)
{
   const struct asahi_ccmd_gem_bind_req *req = to_asahi_ccmd_gem_bind_req(hdr);
   struct asahi_object *obj = asahi_get_object_from_res_id(actx, req->res_id);
   int ret = 0;

   if (!obj) {
      drm_log("Could not lookup obj: res_id=%u", req->res_id);
      return -ENOENT;
   }

   drm_dbg("gem_bind: handle=%d\n", obj->handle);

   struct drm_asahi_gem_bind gem_bind = {
      .op = req->op,
      .flags = req->flags,
      .handle = obj->handle,
      .vm_id = req->vm_id,
      .offset = 0,
      .range = req->size,
      .addr = req->addr,
   };

   ret = drmIoctl(actx->fd, DRM_IOCTL_ASAHI_GEM_BIND, &gem_bind);
   if (ret) {
      drm_log("DRM_IOCTL_ASAHI_GEM_BIND failed: (handle=%d)\n", obj->handle);
   }

   return ret;
}

static int
asahi_ccmd_submit(struct asahi_context *actx, const struct vdrm_ccmd_req *hdr)
{
   const struct asahi_ccmd_submit_req *req = to_asahi_ccmd_submit_req(hdr);
   int ret = 0;

   drm_log("number of commands: %d", req->command_count);

   if (req->command_count == 0) {
      return -EINVAL;
   }

   struct drm_asahi_command *commands = (struct drm_asahi_command *)calloc(
      req->command_count, sizeof(struct drm_asahi_command));

   if (hdr->len < sizeof(struct asahi_ccmd_submit_req)) {
      return -EINVAL;
   }

   uint64_t payload_end =
      (uint64_t)&req->payload + (hdr->len - sizeof(struct asahi_ccmd_submit_req));

   char *ptr = (char *)req->payload;
   for (uint32_t i = 0; i < req->command_count; i++) {
      struct drm_asahi_command *cmd = (struct drm_asahi_command *)ptr;
      if (((uint64_t)cmd + sizeof(struct drm_asahi_command)) > payload_end) {
         ret = -EINVAL;
         goto free_cmd;
      }
      memcpy(&commands[i], cmd, sizeof(struct drm_asahi_command));

      uint64_t cmd_buffer = (uint64_t)(uintptr_t)ptr + sizeof(struct drm_asahi_command);
      commands[i].cmd_buffer = cmd_buffer;

      ptr += sizeof(struct drm_asahi_command);

      if (((uint64_t)ptr + commands[i].cmd_buffer_size) > payload_end) {
         ret = -EINVAL;
         goto free_cmd;
      }
      ptr += commands[i].cmd_buffer_size;

      if (commands[i].cmd_type == DRM_ASAHI_CMD_RENDER) {
         struct drm_asahi_cmd_render *c = (struct drm_asahi_cmd_render *)cmd_buffer;
         drm_dbg("command is RENDER: fragments = %d", c->fragment_attachment_count);

         uint64_t attachments_size;
         uint64_t attachments_end;
         if (__builtin_mul_overflow(c->fragment_attachment_count,
                                    sizeof(struct drm_asahi_attachment),
                                    &attachments_size)) {
            ret = -EINVAL;
            goto free_cmd;
         }
         if (__builtin_add_overflow(cmd_buffer + commands[i].cmd_buffer_size,
                                    attachments_size, &attachments_end)) {
            ret = -EINVAL;
            goto free_cmd;
         }
         if (attachments_end > payload_end) {
            ret = -EINVAL;
            goto free_cmd;
         }

         c->fragment_attachments = cmd_buffer + commands[i].cmd_buffer_size;
         ptr += attachments_size;
      } else if (commands[i].cmd_type == DRM_ASAHI_CMD_COMPUTE) {
         drm_dbg("command is COMPUTE");
      } else {
         drm_log("Unknown command: %d", commands[i].cmd_type);
      }
   }

   struct drm_asahi_submit submit = {
      .flags = 0,
      .queue_id = req->queue_id,
      .result_handle = handle_from_res_id(actx, req->result_res_id),
      .command_count = req->command_count,
      .commands = (uint64_t)(uintptr_t)&commands[0],
   };

   struct drm_asahi_sync in_sync = { .sync_type = DRM_ASAHI_SYNC_SYNCOBJ };
   int in_fence_fd = virgl_context_take_in_fence_fd(&actx->base);

   if (in_fence_fd >= 0) {
      ret = drmSyncobjCreate(actx->fd, 0, &in_sync.handle);
      assert(ret == 0);
      ret = drmSyncobjImportSyncFile(actx->fd, in_sync.handle, in_fence_fd);
      if (ret == 0) {
         submit.in_sync_count = 1;
         submit.in_syncs = (uint64_t)(uintptr_t)&in_sync;
      }
   }

   struct drm_asahi_sync out_sync = { .sync_type = DRM_ASAHI_SYNC_SYNCOBJ };

   ret = drmSyncobjCreate(actx->fd, 0, &out_sync.handle);
   if (ret == 0) {
      submit.out_sync_count = 1;
      submit.out_syncs = (uint64_t)(uintptr_t)&out_sync;
   } else {
      drm_dbg("out syncobj creation failed");
   }

   ret = drmIoctl(actx->fd, DRM_IOCTL_ASAHI_SUBMIT, &submit);
   if (ret) {
      drm_log("DRM_IOCTL_ASAHI_SUBMIT failed: %d", ret);
   }

   if (in_fence_fd >= 0) {
      close(in_fence_fd);
      drmSyncobjDestroy(actx->fd, in_sync.handle);
   }

   if (ret == 0) {
      int submit_fd;
      ret = drmSyncobjExportSyncFile(actx->fd, out_sync.handle, &submit_fd);
      if (ret == 0) {
         drm_timeline_set_last_fence_fd(&actx->timelines[0], submit_fd);
         drm_dbg("set last fd ring_idx: %d", submit_fd);
      } else {
         drm_log("failed to create a FD from the syncobj (%d)", ret);
      }
   } else {
      drm_log("command submission failed");
   }

   drmSyncobjDestroy(actx->fd, out_sync.handle);
free_cmd:
   free(commands);

   return ret;
}

static const struct ccmd {
   const char *name;
   int (*handler)(struct asahi_context *actx, const struct vdrm_ccmd_req *hdr);
   size_t size;
} ccmd_dispatch[] = {
#define HANDLER(N, n)                                                                    \
   [ASAHI_CCMD_##N] = { #N, asahi_ccmd_##n, sizeof(struct asahi_ccmd_##n##_req) }
   HANDLER(NOP, nop),
   HANDLER(IOCTL_SIMPLE, ioctl_simple),
   HANDLER(GET_PARAMS, get_params),
   HANDLER(GEM_NEW, gem_new),
   HANDLER(GEM_BIND, gem_bind),
   HANDLER(SUBMIT, submit),
};

static int
submit_cmd_dispatch(struct asahi_context *actx, const struct vdrm_ccmd_req *hdr)
{
   int ret;

   if (hdr->cmd >= ARRAY_SIZE(ccmd_dispatch)) {
      drm_log("invalid cmd: %u", hdr->cmd);
      return -EINVAL;
   }

   const struct ccmd *ccmd = &ccmd_dispatch[hdr->cmd];

   if (!ccmd->handler) {
      drm_log("no handler: %u", hdr->cmd);
      return -EINVAL;
   }

   drm_dbg("%s: hdr={cmd=%u, len=%u, seqno=%u, rsp_off=0x%x)", ccmd->name, hdr->cmd,
           hdr->len, hdr->seqno, hdr->rsp_off);

   /* If the request length from the guest is smaller than the expected
    * size, ie. newer host and older guest, we need to make a copy of
    * the request with the new fields at the end zero initialized.
    */
   if (ccmd->size > hdr->len) {
      uint8_t buf[ccmd->size];

      memcpy(&buf[0], hdr, hdr->len);
      memset(&buf[hdr->len], 0, ccmd->size - hdr->len);

      ret = ccmd->handler(actx, (struct vdrm_ccmd_req *)buf);
   } else {
      ret = ccmd->handler(actx, hdr);
   }

   if (ret) {
      drm_log("%s: dispatch failed: %d (%s)", ccmd->name, ret, strerror(errno));
      return ret;
   }

   /* If the response length from the guest is smaller than the
    * expected size, ie. newer host and older guest, then a shadow
    * copy is used, and we need to copy back to the actual rsp
    * buffer.
    */
   struct vdrm_ccmd_rsp *rsp = asahi_context_rsp_noshadow(actx, hdr);
   if (actx->current_rsp && (actx->current_rsp != rsp)) {
      unsigned len = rsp->len;
      memcpy(rsp, actx->current_rsp, len);
      rsp->len = len;
      free(actx->current_rsp);
   }
   actx->current_rsp = NULL;

   /* Note that commands with no response, like SET_DEBUGINFO, could
    * be sent before the shmem buffer is allocated:
    */
   if (actx->shmem) {
      /* TODO better way to do this?  We need ACQ_REL semanatics (AFAIU)
       * to ensure that writes to response buffer are visible to the
       * guest process before the update of the seqno.  Otherwise we
       * could just use p_atomic_set.
       */
      uint32_t seqno = hdr->seqno;
      drm_log("updating seqno=%d\n", seqno);
      p_atomic_xchg(&actx->shmem->base.seqno, seqno);
   }

   return 0;
}

static int
asahi_renderer_submit_cmd(struct virgl_context *vctx, const void *_buffer, size_t size)
{
   struct asahi_context *actx = to_asahi_context(vctx);
   const uint8_t *buffer = _buffer;

   while (size >= sizeof(struct vdrm_ccmd_req)) {
      const struct vdrm_ccmd_req *hdr = (const struct vdrm_ccmd_req *)buffer;

      /* Sanity check first: */
      if ((hdr->len > size) || (hdr->len < sizeof(*hdr)) || (hdr->len % 4)) {
         drm_log("bad size, %u vs %zu (%u)", hdr->len, size, hdr->cmd);
         goto cont;
      }

      if (hdr->rsp_off % 4) {
         drm_log("bad rsp_off, %u", hdr->rsp_off);
         goto cont;
      }

      int ret = submit_cmd_dispatch(actx, hdr);
      if (ret) {
         drm_log("dispatch failed: %d (%u)", ret, hdr->cmd);
      }

   cont:
      buffer += hdr->len;
      size -= hdr->len;
   }

   if (size > 0) {
      drm_log("bad size, %zu trailing bytes", size);
      return -EINVAL;
   }

   return 0;
}

static int
asahi_renderer_get_fencing_fd(struct virgl_context *vctx)
{
   struct asahi_context *actx = to_asahi_context(vctx);
   return actx->eventfd;
}

static void
asahi_renderer_retire_fences(UNUSED struct virgl_context *vctx)
{
   /* No-op as VIRGL_RENDERER_ASYNC_FENCE_CB is required */
}

static void
asahi_renderer_fence_retire(struct virgl_context *vctx, uint32_t ring_idx,
                            uint64_t fence_id)
{
   vctx->fence_retire(vctx, ring_idx, fence_id);
}

static int
asahi_renderer_submit_fence(struct virgl_context *vctx, uint32_t flags, uint32_t ring_idx,
                            uint64_t fence_id)
{
   struct asahi_context *actx = to_asahi_context(vctx);

   drm_dbg("flags=0x%x, ring_idx=%" PRIu32 ", fence_id=%" PRIu64, flags, ring_idx,
           fence_id);

   /* ring_idx zero is used for the guest to synchronize with host CPU,
    * meaning by the time ->submit_fence() is called, the fence has
    * already passed.. so just immediate signal:
    */
   if (ring_idx == 0) {
      vctx->fence_retire(vctx, ring_idx, fence_id);
      return 0;
   }

   return drm_timeline_submit_fence(&actx->timelines[0], flags, fence_id);
}

struct virgl_context *
asahi_renderer_create(int fd, UNUSED size_t debug_len, UNUSED const char *debug_name)
{
   struct asahi_context *actx;

   drm_log("");

   actx = calloc(1, sizeof(*actx) + (nr_timelines * sizeof(actx->timelines[0])));
   if (!actx)
      return NULL;

   actx->fd = fd;

   /* Indexed by blob_id, but only lower 32b of blob_id are used: */
   actx->blob_table = _mesa_hash_table_create_u32_keys(NULL);
   /* Indexed by res_id: */
   actx->resource_table = _mesa_hash_table_create_u32_keys(NULL);

   actx->eventfd = create_eventfd(0);

   drm_timeline_init(&actx->timelines[0], &actx->base, "asahi-sync", actx->eventfd, 1,
                     asahi_renderer_fence_retire);

   actx->base.destroy = asahi_renderer_destroy;
   actx->base.attach_resource = asahi_renderer_attach_resource;
   actx->base.detach_resource = asahi_renderer_detach_resource;
   actx->base.export_opaque_handle = asahi_renderer_export_opaque_handle;
   actx->base.transfer_3d = asahi_renderer_transfer_3d;
   actx->base.get_blob = asahi_renderer_get_blob;
   actx->base.submit_cmd = asahi_renderer_submit_cmd;
   actx->base.get_fencing_fd = asahi_renderer_get_fencing_fd;
   actx->base.retire_fences = asahi_renderer_retire_fences;
   actx->base.submit_fence = asahi_renderer_submit_fence;
   actx->base.resource_map = asahi_renderer_resource_map;
   actx->base.supports_fence_sharing = true;

   return &actx->base;
}
