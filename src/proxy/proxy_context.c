/*
 * Copyright 2021 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "proxy_context.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <unistd.h>

#include "server/render_protocol.h"
#include "util/anon_file.h"
#include "util/bitscan.h"

#include "proxy_client.h"

struct proxy_fence {
   uint32_t flags;
   uint32_t seqno;
   void *cookie;
   struct list_head head;
};

static bool
proxy_fence_is_signaled(const struct proxy_fence *fence, uint32_t cur_seqno)
{
   /* takes wrapping into account */
   const uint32_t d = cur_seqno - fence->seqno;
   return d < INT32_MAX;
}

static struct proxy_fence *
proxy_context_alloc_fence(struct proxy_context *ctx)
{
   struct proxy_fence *fence = NULL;

   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_lock(&ctx->free_fences_mutex);

   if (!list_is_empty(&ctx->free_fences)) {
      fence = list_first_entry(&ctx->free_fences, struct proxy_fence, head);
      list_del(&fence->head);
   }

   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_unlock(&ctx->free_fences_mutex);

   return fence ? fence : malloc(sizeof(*fence));
}

static void
proxy_context_free_fence(struct proxy_context *ctx, struct proxy_fence *fence)
{
   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_lock(&ctx->free_fences_mutex);

   list_add(&fence->head, &ctx->free_fences);

   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_unlock(&ctx->free_fences_mutex);
}

static uint32_t
proxy_context_load_timeline_seqno(struct proxy_context *ctx, uint32_t ring_idx)
{
   return atomic_load(&ctx->timeline_seqnos[ring_idx]);
}

static bool
proxy_context_retire_timeline_fences_locked(struct proxy_context *ctx,
                                            uint32_t ring_idx,
                                            uint32_t cur_seqno)
{
   struct proxy_timeline *timeline = &ctx->timelines[ring_idx];
   bool force_retire_all = false;

   /* check if the socket has been disconnected (i.e., the other end has
    * crashed) if no progress is made after a while
    */
   if (timeline->cur_seqno == cur_seqno && !list_is_empty(&timeline->fences)) {
      timeline->cur_seqno_stall_count++;
      if (timeline->cur_seqno_stall_count < 100 ||
          proxy_socket_is_connected(&ctx->socket))
         return false;

      /* socket has been disconnected */
      force_retire_all = true;
   }

   timeline->cur_seqno = cur_seqno;
   timeline->cur_seqno_stall_count = 0;

   list_for_each_entry_safe (struct proxy_fence, fence, &timeline->fences, head) {
      if (!proxy_fence_is_signaled(fence, timeline->cur_seqno) && !force_retire_all)
         return false;

      ctx->base.fence_retire(&ctx->base, ring_idx, fence->cookie);

      list_del(&fence->head);
      proxy_context_free_fence(ctx, fence);
   }

   return true;
}

static void
proxy_context_retire_fences_internal(struct proxy_context *ctx)
{
   if (ctx->sync_thread.fence_eventfd >= 0)
      flush_eventfd(ctx->sync_thread.fence_eventfd);

   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_lock(&ctx->timeline_mutex);

   uint64_t new_busy_mask = 0;
   uint64_t old_busy_mask = ctx->timeline_busy_mask;
   while (old_busy_mask) {
      const uint32_t ring_idx = u_bit_scan64(&old_busy_mask);
      const uint32_t cur_seqno = proxy_context_load_timeline_seqno(ctx, ring_idx);
      if (!proxy_context_retire_timeline_fences_locked(ctx, ring_idx, cur_seqno))
         new_busy_mask |= 1 << ring_idx;
   }

   ctx->timeline_busy_mask = new_busy_mask;

   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_unlock(&ctx->timeline_mutex);
}

static int
proxy_context_sync_thread(void *arg)
{
   struct proxy_context *ctx = arg;
   struct pollfd poll_fds[2] = {
      [0] = {
         .fd = ctx->sync_thread.fence_eventfd,
         .events = POLLIN,
      },
      [1] = {
         .fd = ctx->socket.fd,
      },
   };

   assert(proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB);

   while (!ctx->sync_thread.stop) {
      const int ret = poll(poll_fds, ARRAY_SIZE(poll_fds), -1);
      if (ret <= 0) {
         if (ret < 0 && (errno == EINTR || errno == EAGAIN))
            continue;

         proxy_log("failed to poll fence eventfd");
         break;
      }

      proxy_context_retire_fences_internal(ctx);
   }

   return 0;
}

static int
proxy_context_submit_fence(struct virgl_context *base,
                           uint32_t flags,
                           uint64_t queue_id,
                           void *fence_cookie)
{
   struct proxy_context *ctx = (struct proxy_context *)base;

   /* TODO fix virglrenderer to match virtio-gpu spec which uses ring_idx */
   const uint32_t ring_idx = queue_id;
   if (ring_idx >= PROXY_CONTEXT_TIMELINE_COUNT)
      return -EINVAL;

   struct proxy_fence *fence = proxy_context_alloc_fence(ctx);
   if (!fence)
      return -ENOMEM;

   struct proxy_timeline *timeline = &ctx->timelines[ring_idx];
   const uint32_t seqno = timeline->next_seqno++;
   const struct render_context_op_submit_fence_request req = {
      .header.op = RENDER_CONTEXT_OP_SUBMIT_FENCE,
      .flags = flags,
      .ring_index = ring_idx,
      .seqno = seqno,
   };
   if (!proxy_socket_send_request(&ctx->socket, &req, sizeof(req))) {
      proxy_log("failed to submit fence");
      proxy_context_free_fence(ctx, fence);
      return -1;
   }

   fence->flags = flags;
   fence->seqno = seqno;
   fence->cookie = fence_cookie;

   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_lock(&ctx->timeline_mutex);

   list_addtail(&fence->head, &timeline->fences);
   ctx->timeline_busy_mask |= 1 << ring_idx;

   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      mtx_unlock(&ctx->timeline_mutex);

   return 0;
}

static void
proxy_context_retire_fences(struct virgl_context *base)
{
   struct proxy_context *ctx = (struct proxy_context *)base;

   assert(!(proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB));
   proxy_context_retire_fences_internal(ctx);
}

static int
proxy_context_get_fencing_fd(struct virgl_context *base)
{
   struct proxy_context *ctx = (struct proxy_context *)base;

   assert(!(proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB));
   return ctx->sync_thread.fence_eventfd;
}

static int
proxy_context_submit_cmd(struct virgl_context *base, const void *buffer, size_t size)
{
   struct proxy_context *ctx = (struct proxy_context *)base;

   if (!size)
      return 0;

   struct render_context_op_submit_cmd_request req = {
      .header.op = RENDER_CONTEXT_OP_SUBMIT_CMD,
      .size = size,
   };

   const size_t inlined = MIN2(size, sizeof(req.cmd));
   memcpy(req.cmd, buffer, inlined);

   if (!proxy_socket_send_request(&ctx->socket, &req, sizeof(req))) {
      proxy_log("failed to submit cmd");
      return -1;
   }

   if (size > inlined) {
      if (!proxy_socket_send_request(&ctx->socket, (const char *)buffer + inlined,
                                     size - inlined)) {
         proxy_log("failed to submit large cmd buffer");
         return -1;
      }
   }

   /* XXX this is forced a roundtrip to avoid surprises; vtest requires this
    * at least
    */
   struct render_context_op_submit_cmd_reply reply;
   if (!proxy_socket_receive_reply(&ctx->socket, &reply, sizeof(reply))) {
      proxy_log("failed to get submit result");
      return -1;
   }

   return reply.ok ? 0 : -1;
}

static int
alloc_memfd(const char *name, size_t size, void **out_ptr)
{
   int fd = os_create_anonymous_file(size, name);
   if (fd < 0)
      return -1;

   int ret = fcntl(fd, F_ADD_SEALS, F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW);
   if (ret)
      goto fail;

   if (!out_ptr)
      return fd;

   void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
   if (ptr == MAP_FAILED)
      goto fail;

   *out_ptr = ptr;
   return fd;

fail:
   close(fd);
   return -1;
}

static int
proxy_context_get_blob(struct virgl_context *base,
                       UNUSED uint32_t res_id,
                       uint64_t blob_id,
                       uint64_t blob_size,
                       uint32_t blob_flags,
                       struct virgl_context_blob *blob)
{
   struct proxy_context *ctx = (struct proxy_context *)base;

   /* hijack blob_id == 0 && blob_flags == MMAPABLE to save roundtrips */
   if (!blob_id && blob_flags == VIRGL_RENDERER_BLOB_FLAG_USE_MAPPABLE) {
      int fd = alloc_memfd("proxy-blob", blob_size, NULL);
      if (fd < 0)
         return -ENOMEM;

      blob->type = VIRGL_RESOURCE_FD_SHM;
      blob->u.fd = fd;
      blob->map_info = VIRGL_RENDERER_MAP_CACHE_CACHED;
      return 0;
   }

   const struct render_context_op_get_blob_request req = {
      .header.op = RENDER_CONTEXT_OP_GET_BLOB,
      .blob_id = blob_id,
      .blob_size = blob_size,
      .blob_flags = blob_flags,
   };
   if (!proxy_socket_send_request(&ctx->socket, &req, sizeof(req))) {
      proxy_log("failed to get blob %" PRIu64, blob_id);
      return -1;
   }

   struct render_context_op_get_blob_reply reply;
   int reply_fd;
   int reply_fd_count;
   if (!proxy_socket_receive_reply_with_fds(&ctx->socket, &reply, sizeof(reply),
                                            &reply_fd, 1, &reply_fd_count)) {
      proxy_log("failed to get reply of blob %" PRIu64, blob_id);
      return -1;
   }

   if (!reply_fd_count) {
      proxy_log("invalid reply for blob %" PRIu64, blob_id);
      return -1;
   }

   bool reply_fd_valid = false;
   switch (reply.fd_type) {
   case VIRGL_RESOURCE_FD_DMABUF:
      /* TODO validate the fd is dmabuf >= blob_size */
      reply_fd_valid = true;
      break;
   case VIRGL_RESOURCE_FD_OPAQUE:
      /* this will be validated when imported by the client */
      reply_fd_valid = true;
      break;
   case VIRGL_RESOURCE_FD_SHM:
      /* we don't expect shm, otherwise we should validate seals and size */
      reply_fd_valid = false;
      break;
   default:
      break;
   }
   if (!reply_fd_valid) {
      proxy_log("invalid fd type %d for blob %" PRIu64, reply.fd_type, blob_id);
      close(reply_fd);
      return -1;
   }

   blob->type = reply.fd_type;
   blob->u.fd = reply_fd;
   blob->map_info = reply.map_info;

   return 0;
}

static int
proxy_context_transfer_3d(struct virgl_context *base,
                          struct virgl_resource *res,
                          UNUSED const struct vrend_transfer_info *info,
                          UNUSED int transfer_mode)
{
   struct proxy_context *ctx = (struct proxy_context *)base;

   proxy_log("no transfer support for ctx %d and res %d", ctx->base.ctx_id, res->res_id);
   return -1;
}

static void
proxy_context_detach_resource(struct virgl_context *base, struct virgl_resource *res)
{
   struct proxy_context *ctx = (struct proxy_context *)base;

   const struct render_context_op_detach_resource_request req = {
      .header.op = RENDER_CONTEXT_OP_DETACH_RESOURCE,
      .res_id = res->res_id,
   };
   if (!proxy_socket_send_request(&ctx->socket, &req, sizeof(req)))
      proxy_log("failed to detach res %d", res->res_id);
}

static void
proxy_context_attach_resource(struct virgl_context *base, struct virgl_resource *res)
{
   struct proxy_context *ctx = (struct proxy_context *)base;

   enum virgl_resource_fd_type res_fd_type = res->fd_type;
   int res_fd = res->fd;
   bool close_res_fd = false;
   if (res_fd_type == VIRGL_RESOURCE_FD_INVALID) {
      res_fd_type = virgl_resource_export_fd(res, &res_fd);
      if (res_fd_type == VIRGL_RESOURCE_FD_INVALID) {
         proxy_log("failed to export res %d", res->res_id);
         return;
      }

      close_res_fd = true;
   }

   /* the proxy ignores iovs since transfer_3d is not supported */
   const struct render_context_op_attach_resource_request req = {
      .header.op = RENDER_CONTEXT_OP_ATTACH_RESOURCE,
      .res_id = res->res_id,
      .fd_type = res_fd_type,
      .size = virgl_resource_get_size(res),
   };
   if (!proxy_socket_send_request_with_fds(&ctx->socket, &req, sizeof(req), &res_fd, 1))
      proxy_log("failed to attach res %d", res->res_id);

   if (res_fd >= 0 && close_res_fd)
      close(res_fd);
}

static void
proxy_context_destroy(struct virgl_context *base)
{
   struct proxy_context *ctx = (struct proxy_context *)base;

   /* ask the server process to terminate the context process */
   if (!proxy_client_destroy_context(ctx->client, ctx->base.ctx_id))
      proxy_log("failed to destroy ctx %d", ctx->base.ctx_id);

   if (ctx->sync_thread.fence_eventfd >= 0) {
      if (ctx->sync_thread.created) {
         ctx->sync_thread.stop = true;
         write_eventfd(ctx->sync_thread.fence_eventfd, 1);
         thrd_join(ctx->sync_thread.thread, NULL);
      }

      close(ctx->sync_thread.fence_eventfd);
   }

   if (ctx->shmem.ptr)
      munmap(ctx->shmem.ptr, ctx->shmem.size);
   if (ctx->shmem.fd >= 0)
      close(ctx->shmem.fd);

   if (ctx->timeline_seqnos) {
      for (uint32_t i = 0; i < PROXY_CONTEXT_TIMELINE_COUNT; i++) {
         struct proxy_timeline *timeline = &ctx->timelines[i];
         list_for_each_entry_safe (struct proxy_fence, fence, &timeline->fences, head)
            free(fence);
      }
   }
   mtx_destroy(&ctx->timeline_mutex);

   list_for_each_entry_safe (struct proxy_fence, fence, &ctx->free_fences, head)
      free(fence);
   mtx_destroy(&ctx->free_fences_mutex);

   proxy_socket_fini(&ctx->socket);

   free(ctx);
}

static void
proxy_context_init_base(struct proxy_context *ctx)
{
   ctx->base.destroy = proxy_context_destroy;
   ctx->base.attach_resource = proxy_context_attach_resource;
   ctx->base.detach_resource = proxy_context_detach_resource;
   ctx->base.transfer_3d = proxy_context_transfer_3d;
   ctx->base.get_blob = proxy_context_get_blob;
   ctx->base.submit_cmd = proxy_context_submit_cmd;

   ctx->base.get_fencing_fd = proxy_context_get_fencing_fd;
   ctx->base.retire_fences = proxy_context_retire_fences;
   ctx->base.submit_fence = proxy_context_submit_fence;
}

static bool
proxy_context_init_fencing(struct proxy_context *ctx)
{
   /* The render server updates the shmem for the current seqnos and
    * optionally notifies using the eventfd.  That means, when only
    * VIRGL_RENDERER_THREAD_SYNC is set, we just need to set up the eventfd.
    * When VIRGL_RENDERER_ASYNC_FENCE_CB is also set, we need to create a sync
    * thread as well.
    *
    * Fence polling can always check the shmem directly.
    */
   if (!(proxy_renderer.flags & VIRGL_RENDERER_THREAD_SYNC))
      return true;

   ctx->sync_thread.fence_eventfd = create_eventfd(0);
   if (ctx->sync_thread.fence_eventfd < 0) {
      proxy_log("failed to create fence eventfd");
      return false;
   }

   if (proxy_renderer.flags & VIRGL_RENDERER_ASYNC_FENCE_CB) {
      int ret = thrd_create(&ctx->sync_thread.thread, proxy_context_sync_thread, ctx);
      if (ret != thrd_success) {
         proxy_log("failed to create sync thread");
         return false;
      }
      ctx->sync_thread.created = true;
   }

   return true;
}

static bool
proxy_context_init_timelines(struct proxy_context *ctx)
{
   atomic_uint *timeline_seqnos = ctx->shmem.ptr;
   for (uint32_t i = 0; i < ARRAY_SIZE(ctx->timelines); i++) {
      atomic_init(&timeline_seqnos[i], 0);

      struct proxy_timeline *timeline = &ctx->timelines[i];
      timeline->cur_seqno = 0;
      timeline->next_seqno = 1;
      list_inithead(&timeline->fences);
   }

   ctx->timeline_seqnos = timeline_seqnos;

   return true;
}

static bool
proxy_context_init_shmem(struct proxy_context *ctx)
{
   const size_t shmem_size = sizeof(*ctx->timeline_seqnos) * PROXY_CONTEXT_TIMELINE_COUNT;
   ctx->shmem.fd = alloc_memfd("proxy-ctx", shmem_size, &ctx->shmem.ptr);
   if (ctx->shmem.fd < 0)
      return false;

   ctx->shmem.size = shmem_size;

   return true;
}

static bool
proxy_context_init(struct proxy_context *ctx, uint32_t ctx_flags)
{
   if (!proxy_context_init_shmem(ctx) || !proxy_context_init_timelines(ctx) ||
       !proxy_context_init_fencing(ctx))
      return false;

   const struct render_context_op_init_request req = {
      .header.op = RENDER_CONTEXT_OP_INIT,
      .flags = ctx_flags,
      .shmem_size = ctx->shmem.size,
   };
   const int req_fds[2] = { ctx->shmem.fd, ctx->sync_thread.fence_eventfd };
   const int req_fd_count = req_fds[1] >= 0 ? 2 : 1;
   if (!proxy_socket_send_request_with_fds(&ctx->socket, &req, sizeof(req), req_fds,
                                           req_fd_count)) {
      proxy_log("failed to initialize context");
      return false;
   }

   return true;
}

struct virgl_context *
proxy_context_create(uint32_t ctx_id,
                     uint32_t ctx_flags,
                     size_t debug_len,
                     const char *debug_name)
{
   struct proxy_client *client = proxy_renderer.client;
   struct proxy_context *ctx;

   int ctx_fd;
   if (!proxy_client_create_context(client, ctx_id, debug_len, debug_name, &ctx_fd)) {
      proxy_log("failed to create a context");
      return NULL;
   }

   ctx = calloc(1, sizeof(*ctx));
   if (!ctx) {
      close(ctx_fd);
      return NULL;
   }

   proxy_context_init_base(ctx);
   ctx->client = client;
   proxy_socket_init(&ctx->socket, ctx_fd);
   ctx->shmem.fd = -1;
   mtx_init(&ctx->timeline_mutex, mtx_plain);
   mtx_init(&ctx->free_fences_mutex, mtx_plain);
   list_inithead(&ctx->free_fences);
   ctx->sync_thread.fence_eventfd = -1;

   if (!proxy_context_init(ctx, ctx_flags)) {
      proxy_context_destroy(&ctx->base);
      return NULL;
   }

   return &ctx->base;
}
