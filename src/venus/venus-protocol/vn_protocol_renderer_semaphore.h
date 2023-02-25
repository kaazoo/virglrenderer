/* This file is generated by venus-protocol.  See vn_protocol_renderer.h. */

/*
 * Copyright 2020 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef VN_PROTOCOL_RENDERER_SEMAPHORE_H
#define VN_PROTOCOL_RENDERER_SEMAPHORE_H

#include "vn_protocol_renderer_structs.h"

#pragma GCC diagnostic push
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 12
#pragma GCC diagnostic ignored "-Wdangling-pointer"
#endif
#pragma GCC diagnostic ignored "-Wpointer-arith"
#pragma GCC diagnostic ignored "-Wunused-parameter"

/*
 * These structs/unions/commands are not included
 *
 *   vkGetSemaphoreFdKHR
 *   vkImportSemaphoreFdKHR
 */

/* struct VkExportSemaphoreCreateInfo chain */

static inline void *
vn_decode_VkExportSemaphoreCreateInfo_pnext_temp(struct vn_cs_decoder *dec)
{
    /* no known/supported struct */
    if (vn_decode_simple_pointer(dec))
        vn_cs_decoder_set_fatal(dec);
    return NULL;
}

static inline void
vn_decode_VkExportSemaphoreCreateInfo_self_temp(struct vn_cs_decoder *dec, VkExportSemaphoreCreateInfo *val)
{
    /* skip val->{sType,pNext} */
    vn_decode_VkFlags(dec, &val->handleTypes);
}

static inline void
vn_decode_VkExportSemaphoreCreateInfo_temp(struct vn_cs_decoder *dec, VkExportSemaphoreCreateInfo *val)
{
    VkStructureType stype;
    vn_decode_VkStructureType(dec, &stype);
    if (stype != VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO)
        vn_cs_decoder_set_fatal(dec);

    val->sType = stype;
    val->pNext = vn_decode_VkExportSemaphoreCreateInfo_pnext_temp(dec);
    vn_decode_VkExportSemaphoreCreateInfo_self_temp(dec, val);
}

static inline void
vn_replace_VkExportSemaphoreCreateInfo_handle_self(VkExportSemaphoreCreateInfo *val)
{
    /* skip val->sType */
    /* skip val->pNext */
    /* skip val->handleTypes */
}

static inline void
vn_replace_VkExportSemaphoreCreateInfo_handle(VkExportSemaphoreCreateInfo *val)
{
    struct VkBaseOutStructure *pnext = (struct VkBaseOutStructure *)val;

    do {
        switch ((int32_t)pnext->sType) {
        case VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO:
            vn_replace_VkExportSemaphoreCreateInfo_handle_self((VkExportSemaphoreCreateInfo *)pnext);
            break;
        default:
            /* ignore unknown/unsupported struct */
            break;
        }
        pnext = pnext->pNext;
    } while (pnext);
}

/* struct VkSemaphoreCreateInfo chain */

static inline void *
vn_decode_VkSemaphoreCreateInfo_pnext_temp(struct vn_cs_decoder *dec)
{
    VkBaseOutStructure *pnext;
    VkStructureType stype;

    if (!vn_decode_simple_pointer(dec))
        return NULL;

    vn_decode_VkStructureType(dec, &stype);
    switch ((int32_t)stype) {
    case VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO:
        pnext = vn_cs_decoder_alloc_temp(dec, sizeof(VkExportSemaphoreCreateInfo));
        if (pnext) {
            pnext->sType = stype;
            pnext->pNext = vn_decode_VkSemaphoreCreateInfo_pnext_temp(dec);
            vn_decode_VkExportSemaphoreCreateInfo_self_temp(dec, (VkExportSemaphoreCreateInfo *)pnext);
        }
        break;
    case VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO:
        pnext = vn_cs_decoder_alloc_temp(dec, sizeof(VkSemaphoreTypeCreateInfo));
        if (pnext) {
            pnext->sType = stype;
            pnext->pNext = vn_decode_VkSemaphoreCreateInfo_pnext_temp(dec);
            vn_decode_VkSemaphoreTypeCreateInfo_self_temp(dec, (VkSemaphoreTypeCreateInfo *)pnext);
        }
        break;
    default:
        /* unexpected struct */
        pnext = NULL;
        vn_cs_decoder_set_fatal(dec);
        break;
    }

    return pnext;
}

static inline void
vn_decode_VkSemaphoreCreateInfo_self_temp(struct vn_cs_decoder *dec, VkSemaphoreCreateInfo *val)
{
    /* skip val->{sType,pNext} */
    vn_decode_VkFlags(dec, &val->flags);
}

static inline void
vn_decode_VkSemaphoreCreateInfo_temp(struct vn_cs_decoder *dec, VkSemaphoreCreateInfo *val)
{
    VkStructureType stype;
    vn_decode_VkStructureType(dec, &stype);
    if (stype != VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO)
        vn_cs_decoder_set_fatal(dec);

    val->sType = stype;
    val->pNext = vn_decode_VkSemaphoreCreateInfo_pnext_temp(dec);
    vn_decode_VkSemaphoreCreateInfo_self_temp(dec, val);
}

static inline void
vn_replace_VkSemaphoreCreateInfo_handle_self(VkSemaphoreCreateInfo *val)
{
    /* skip val->sType */
    /* skip val->pNext */
    /* skip val->flags */
}

static inline void
vn_replace_VkSemaphoreCreateInfo_handle(VkSemaphoreCreateInfo *val)
{
    struct VkBaseOutStructure *pnext = (struct VkBaseOutStructure *)val;

    do {
        switch ((int32_t)pnext->sType) {
        case VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO:
            vn_replace_VkSemaphoreCreateInfo_handle_self((VkSemaphoreCreateInfo *)pnext);
            break;
        case VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO:
            vn_replace_VkExportSemaphoreCreateInfo_handle_self((VkExportSemaphoreCreateInfo *)pnext);
            break;
        case VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO:
            vn_replace_VkSemaphoreTypeCreateInfo_handle_self((VkSemaphoreTypeCreateInfo *)pnext);
            break;
        default:
            /* ignore unknown/unsupported struct */
            break;
        }
        pnext = pnext->pNext;
    } while (pnext);
}

/* struct VkSemaphoreWaitInfo chain */

static inline void *
vn_decode_VkSemaphoreWaitInfo_pnext_temp(struct vn_cs_decoder *dec)
{
    /* no known/supported struct */
    if (vn_decode_simple_pointer(dec))
        vn_cs_decoder_set_fatal(dec);
    return NULL;
}

static inline void
vn_decode_VkSemaphoreWaitInfo_self_temp(struct vn_cs_decoder *dec, VkSemaphoreWaitInfo *val)
{
    /* skip val->{sType,pNext} */
    vn_decode_VkFlags(dec, &val->flags);
    vn_decode_uint32_t(dec, &val->semaphoreCount);
    if (vn_peek_array_size(dec)) {
        const uint32_t iter_count = vn_decode_array_size(dec, val->semaphoreCount);
        val->pSemaphores = vn_cs_decoder_alloc_temp(dec, sizeof(*val->pSemaphores) * iter_count);
        if (!val->pSemaphores) return;
        for (uint32_t i = 0; i < iter_count; i++)
            vn_decode_VkSemaphore_lookup(dec, &((VkSemaphore *)val->pSemaphores)[i]);
    } else {
        vn_decode_array_size(dec, val->semaphoreCount);
        val->pSemaphores = NULL;
    }
    if (vn_peek_array_size(dec)) {
        const size_t array_size = vn_decode_array_size(dec, val->semaphoreCount);
        val->pValues = vn_cs_decoder_alloc_temp(dec, sizeof(*val->pValues) * array_size);
        if (!val->pValues) return;
        vn_decode_uint64_t_array(dec, (uint64_t *)val->pValues, array_size);
    } else {
        vn_decode_array_size(dec, val->semaphoreCount);
        val->pValues = NULL;
    }
}

static inline void
vn_decode_VkSemaphoreWaitInfo_temp(struct vn_cs_decoder *dec, VkSemaphoreWaitInfo *val)
{
    VkStructureType stype;
    vn_decode_VkStructureType(dec, &stype);
    if (stype != VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO)
        vn_cs_decoder_set_fatal(dec);

    val->sType = stype;
    val->pNext = vn_decode_VkSemaphoreWaitInfo_pnext_temp(dec);
    vn_decode_VkSemaphoreWaitInfo_self_temp(dec, val);
}

static inline void
vn_replace_VkSemaphoreWaitInfo_handle_self(VkSemaphoreWaitInfo *val)
{
    /* skip val->sType */
    /* skip val->pNext */
    /* skip val->flags */
    /* skip val->semaphoreCount */
    if (val->pSemaphores) {
       for (uint32_t i = 0; i < val->semaphoreCount; i++)
            vn_replace_VkSemaphore_handle(&((VkSemaphore *)val->pSemaphores)[i]);
    }
    /* skip val->pValues */
}

static inline void
vn_replace_VkSemaphoreWaitInfo_handle(VkSemaphoreWaitInfo *val)
{
    struct VkBaseOutStructure *pnext = (struct VkBaseOutStructure *)val;

    do {
        switch ((int32_t)pnext->sType) {
        case VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO:
            vn_replace_VkSemaphoreWaitInfo_handle_self((VkSemaphoreWaitInfo *)pnext);
            break;
        default:
            /* ignore unknown/unsupported struct */
            break;
        }
        pnext = pnext->pNext;
    } while (pnext);
}

/* struct VkSemaphoreSignalInfo chain */

static inline void *
vn_decode_VkSemaphoreSignalInfo_pnext_temp(struct vn_cs_decoder *dec)
{
    /* no known/supported struct */
    if (vn_decode_simple_pointer(dec))
        vn_cs_decoder_set_fatal(dec);
    return NULL;
}

static inline void
vn_decode_VkSemaphoreSignalInfo_self_temp(struct vn_cs_decoder *dec, VkSemaphoreSignalInfo *val)
{
    /* skip val->{sType,pNext} */
    vn_decode_VkSemaphore_lookup(dec, &val->semaphore);
    vn_decode_uint64_t(dec, &val->value);
}

static inline void
vn_decode_VkSemaphoreSignalInfo_temp(struct vn_cs_decoder *dec, VkSemaphoreSignalInfo *val)
{
    VkStructureType stype;
    vn_decode_VkStructureType(dec, &stype);
    if (stype != VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO)
        vn_cs_decoder_set_fatal(dec);

    val->sType = stype;
    val->pNext = vn_decode_VkSemaphoreSignalInfo_pnext_temp(dec);
    vn_decode_VkSemaphoreSignalInfo_self_temp(dec, val);
}

static inline void
vn_replace_VkSemaphoreSignalInfo_handle_self(VkSemaphoreSignalInfo *val)
{
    /* skip val->sType */
    /* skip val->pNext */
    vn_replace_VkSemaphore_handle(&val->semaphore);
    /* skip val->value */
}

static inline void
vn_replace_VkSemaphoreSignalInfo_handle(VkSemaphoreSignalInfo *val)
{
    struct VkBaseOutStructure *pnext = (struct VkBaseOutStructure *)val;

    do {
        switch ((int32_t)pnext->sType) {
        case VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO:
            vn_replace_VkSemaphoreSignalInfo_handle_self((VkSemaphoreSignalInfo *)pnext);
            break;
        default:
            /* ignore unknown/unsupported struct */
            break;
        }
        pnext = pnext->pNext;
    } while (pnext);
}

/* struct VkImportSemaphoreResourceInfo100000MESA chain */

static inline void *
vn_decode_VkImportSemaphoreResourceInfo100000MESA_pnext_temp(struct vn_cs_decoder *dec)
{
    /* no known/supported struct */
    if (vn_decode_simple_pointer(dec))
        vn_cs_decoder_set_fatal(dec);
    return NULL;
}

static inline void
vn_decode_VkImportSemaphoreResourceInfo100000MESA_self_temp(struct vn_cs_decoder *dec, VkImportSemaphoreResourceInfo100000MESA *val)
{
    /* skip val->{sType,pNext} */
    vn_decode_VkSemaphore_lookup(dec, &val->semaphore);
    vn_decode_uint32_t(dec, &val->resourceId);
}

static inline void
vn_decode_VkImportSemaphoreResourceInfo100000MESA_temp(struct vn_cs_decoder *dec, VkImportSemaphoreResourceInfo100000MESA *val)
{
    VkStructureType stype;
    vn_decode_VkStructureType(dec, &stype);
    if (stype != VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_RESOURCE_INFO_100000_MESA)
        vn_cs_decoder_set_fatal(dec);

    val->sType = stype;
    val->pNext = vn_decode_VkImportSemaphoreResourceInfo100000MESA_pnext_temp(dec);
    vn_decode_VkImportSemaphoreResourceInfo100000MESA_self_temp(dec, val);
}

static inline void
vn_replace_VkImportSemaphoreResourceInfo100000MESA_handle_self(VkImportSemaphoreResourceInfo100000MESA *val)
{
    /* skip val->sType */
    /* skip val->pNext */
    vn_replace_VkSemaphore_handle(&val->semaphore);
    /* skip val->resourceId */
}

static inline void
vn_replace_VkImportSemaphoreResourceInfo100000MESA_handle(VkImportSemaphoreResourceInfo100000MESA *val)
{
    struct VkBaseOutStructure *pnext = (struct VkBaseOutStructure *)val;

    do {
        switch ((int32_t)pnext->sType) {
        case VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_RESOURCE_INFO_100000_MESA:
            vn_replace_VkImportSemaphoreResourceInfo100000MESA_handle_self((VkImportSemaphoreResourceInfo100000MESA *)pnext);
            break;
        default:
            /* ignore unknown/unsupported struct */
            break;
        }
        pnext = pnext->pNext;
    } while (pnext);
}

static inline void vn_decode_vkCreateSemaphore_args_temp(struct vn_cs_decoder *dec, struct vn_command_vkCreateSemaphore *args)
{
    vn_decode_VkDevice_lookup(dec, &args->device);
    if (vn_decode_simple_pointer(dec)) {
        args->pCreateInfo = vn_cs_decoder_alloc_temp(dec, sizeof(*args->pCreateInfo));
        if (!args->pCreateInfo) return;
        vn_decode_VkSemaphoreCreateInfo_temp(dec, (VkSemaphoreCreateInfo *)args->pCreateInfo);
    } else {
        args->pCreateInfo = NULL;
        vn_cs_decoder_set_fatal(dec);
    }
    if (vn_decode_simple_pointer(dec)) {
        vn_cs_decoder_set_fatal(dec);
    } else {
        args->pAllocator = NULL;
    }
    if (vn_decode_simple_pointer(dec)) {
        args->pSemaphore = vn_cs_decoder_alloc_temp(dec, sizeof(*args->pSemaphore));
        if (!args->pSemaphore) return;
        vn_decode_VkSemaphore(dec, args->pSemaphore);
    } else {
        args->pSemaphore = NULL;
        vn_cs_decoder_set_fatal(dec);
    }
}

static inline void vn_replace_vkCreateSemaphore_args_handle(struct vn_command_vkCreateSemaphore *args)
{
    vn_replace_VkDevice_handle(&args->device);
    if (args->pCreateInfo)
        vn_replace_VkSemaphoreCreateInfo_handle((VkSemaphoreCreateInfo *)args->pCreateInfo);
    /* skip args->pAllocator */
    /* skip args->pSemaphore */
}

static inline void vn_encode_vkCreateSemaphore_reply(struct vn_cs_encoder *enc, const struct vn_command_vkCreateSemaphore *args)
{
    vn_encode_VkCommandTypeEXT(enc, &(VkCommandTypeEXT){VK_COMMAND_TYPE_vkCreateSemaphore_EXT});

    vn_encode_VkResult(enc, &args->ret);
    /* skip args->device */
    /* skip args->pCreateInfo */
    /* skip args->pAllocator */
    if (vn_encode_simple_pointer(enc, args->pSemaphore))
        vn_encode_VkSemaphore(enc, args->pSemaphore);
}

static inline void vn_decode_vkDestroySemaphore_args_temp(struct vn_cs_decoder *dec, struct vn_command_vkDestroySemaphore *args)
{
    vn_decode_VkDevice_lookup(dec, &args->device);
    vn_decode_VkSemaphore_lookup(dec, &args->semaphore);
    if (vn_decode_simple_pointer(dec)) {
        vn_cs_decoder_set_fatal(dec);
    } else {
        args->pAllocator = NULL;
    }
}

static inline void vn_replace_vkDestroySemaphore_args_handle(struct vn_command_vkDestroySemaphore *args)
{
    vn_replace_VkDevice_handle(&args->device);
    vn_replace_VkSemaphore_handle(&args->semaphore);
    /* skip args->pAllocator */
}

static inline void vn_encode_vkDestroySemaphore_reply(struct vn_cs_encoder *enc, const struct vn_command_vkDestroySemaphore *args)
{
    vn_encode_VkCommandTypeEXT(enc, &(VkCommandTypeEXT){VK_COMMAND_TYPE_vkDestroySemaphore_EXT});

    /* skip args->device */
    /* skip args->semaphore */
    /* skip args->pAllocator */
}

static inline void vn_decode_vkGetSemaphoreCounterValue_args_temp(struct vn_cs_decoder *dec, struct vn_command_vkGetSemaphoreCounterValue *args)
{
    vn_decode_VkDevice_lookup(dec, &args->device);
    vn_decode_VkSemaphore_lookup(dec, &args->semaphore);
    if (vn_decode_simple_pointer(dec)) {
        args->pValue = vn_cs_decoder_alloc_temp(dec, sizeof(*args->pValue));
        if (!args->pValue) return;
    } else {
        args->pValue = NULL;
        vn_cs_decoder_set_fatal(dec);
    }
}

static inline void vn_replace_vkGetSemaphoreCounterValue_args_handle(struct vn_command_vkGetSemaphoreCounterValue *args)
{
    vn_replace_VkDevice_handle(&args->device);
    vn_replace_VkSemaphore_handle(&args->semaphore);
    /* skip args->pValue */
}

static inline void vn_encode_vkGetSemaphoreCounterValue_reply(struct vn_cs_encoder *enc, const struct vn_command_vkGetSemaphoreCounterValue *args)
{
    vn_encode_VkCommandTypeEXT(enc, &(VkCommandTypeEXT){VK_COMMAND_TYPE_vkGetSemaphoreCounterValue_EXT});

    vn_encode_VkResult(enc, &args->ret);
    /* skip args->device */
    /* skip args->semaphore */
    if (vn_encode_simple_pointer(enc, args->pValue))
        vn_encode_uint64_t(enc, args->pValue);
}

static inline void vn_decode_vkWaitSemaphores_args_temp(struct vn_cs_decoder *dec, struct vn_command_vkWaitSemaphores *args)
{
    vn_decode_VkDevice_lookup(dec, &args->device);
    if (vn_decode_simple_pointer(dec)) {
        args->pWaitInfo = vn_cs_decoder_alloc_temp(dec, sizeof(*args->pWaitInfo));
        if (!args->pWaitInfo) return;
        vn_decode_VkSemaphoreWaitInfo_temp(dec, (VkSemaphoreWaitInfo *)args->pWaitInfo);
    } else {
        args->pWaitInfo = NULL;
        vn_cs_decoder_set_fatal(dec);
    }
    vn_decode_uint64_t(dec, &args->timeout);
}

static inline void vn_replace_vkWaitSemaphores_args_handle(struct vn_command_vkWaitSemaphores *args)
{
    vn_replace_VkDevice_handle(&args->device);
    if (args->pWaitInfo)
        vn_replace_VkSemaphoreWaitInfo_handle((VkSemaphoreWaitInfo *)args->pWaitInfo);
    /* skip args->timeout */
}

static inline void vn_encode_vkWaitSemaphores_reply(struct vn_cs_encoder *enc, const struct vn_command_vkWaitSemaphores *args)
{
    vn_encode_VkCommandTypeEXT(enc, &(VkCommandTypeEXT){VK_COMMAND_TYPE_vkWaitSemaphores_EXT});

    vn_encode_VkResult(enc, &args->ret);
    /* skip args->device */
    /* skip args->pWaitInfo */
    /* skip args->timeout */
}

static inline void vn_decode_vkSignalSemaphore_args_temp(struct vn_cs_decoder *dec, struct vn_command_vkSignalSemaphore *args)
{
    vn_decode_VkDevice_lookup(dec, &args->device);
    if (vn_decode_simple_pointer(dec)) {
        args->pSignalInfo = vn_cs_decoder_alloc_temp(dec, sizeof(*args->pSignalInfo));
        if (!args->pSignalInfo) return;
        vn_decode_VkSemaphoreSignalInfo_temp(dec, (VkSemaphoreSignalInfo *)args->pSignalInfo);
    } else {
        args->pSignalInfo = NULL;
        vn_cs_decoder_set_fatal(dec);
    }
}

static inline void vn_replace_vkSignalSemaphore_args_handle(struct vn_command_vkSignalSemaphore *args)
{
    vn_replace_VkDevice_handle(&args->device);
    if (args->pSignalInfo)
        vn_replace_VkSemaphoreSignalInfo_handle((VkSemaphoreSignalInfo *)args->pSignalInfo);
}

static inline void vn_encode_vkSignalSemaphore_reply(struct vn_cs_encoder *enc, const struct vn_command_vkSignalSemaphore *args)
{
    vn_encode_VkCommandTypeEXT(enc, &(VkCommandTypeEXT){VK_COMMAND_TYPE_vkSignalSemaphore_EXT});

    vn_encode_VkResult(enc, &args->ret);
    /* skip args->device */
    /* skip args->pSignalInfo */
}

static inline void vn_decode_vkWaitSemaphoreResource100000MESA_args_temp(struct vn_cs_decoder *dec, struct vn_command_vkWaitSemaphoreResource100000MESA *args)
{
    vn_decode_VkDevice_lookup(dec, &args->device);
    vn_decode_VkSemaphore_lookup(dec, &args->semaphore);
}

static inline void vn_replace_vkWaitSemaphoreResource100000MESA_args_handle(struct vn_command_vkWaitSemaphoreResource100000MESA *args)
{
    vn_replace_VkDevice_handle(&args->device);
    vn_replace_VkSemaphore_handle(&args->semaphore);
}

static inline void vn_encode_vkWaitSemaphoreResource100000MESA_reply(struct vn_cs_encoder *enc, const struct vn_command_vkWaitSemaphoreResource100000MESA *args)
{
    vn_encode_VkCommandTypeEXT(enc, &(VkCommandTypeEXT){VK_COMMAND_TYPE_vkWaitSemaphoreResource100000MESA_EXT});

    /* skip args->device */
    /* skip args->semaphore */
}

static inline void vn_decode_vkImportSemaphoreResource100000MESA_args_temp(struct vn_cs_decoder *dec, struct vn_command_vkImportSemaphoreResource100000MESA *args)
{
    vn_decode_VkDevice_lookup(dec, &args->device);
    if (vn_decode_simple_pointer(dec)) {
        args->pImportSemaphoreResourceInfo = vn_cs_decoder_alloc_temp(dec, sizeof(*args->pImportSemaphoreResourceInfo));
        if (!args->pImportSemaphoreResourceInfo) return;
        vn_decode_VkImportSemaphoreResourceInfo100000MESA_temp(dec, (VkImportSemaphoreResourceInfo100000MESA *)args->pImportSemaphoreResourceInfo);
    } else {
        args->pImportSemaphoreResourceInfo = NULL;
        vn_cs_decoder_set_fatal(dec);
    }
}

static inline void vn_replace_vkImportSemaphoreResource100000MESA_args_handle(struct vn_command_vkImportSemaphoreResource100000MESA *args)
{
    vn_replace_VkDevice_handle(&args->device);
    if (args->pImportSemaphoreResourceInfo)
        vn_replace_VkImportSemaphoreResourceInfo100000MESA_handle((VkImportSemaphoreResourceInfo100000MESA *)args->pImportSemaphoreResourceInfo);
}

static inline void vn_encode_vkImportSemaphoreResource100000MESA_reply(struct vn_cs_encoder *enc, const struct vn_command_vkImportSemaphoreResource100000MESA *args)
{
    vn_encode_VkCommandTypeEXT(enc, &(VkCommandTypeEXT){VK_COMMAND_TYPE_vkImportSemaphoreResource100000MESA_EXT});

    /* skip args->device */
    /* skip args->pImportSemaphoreResourceInfo */
}

static inline void vn_dispatch_vkCreateSemaphore(struct vn_dispatch_context *ctx, VkCommandFlagsEXT flags)
{
    struct vn_command_vkCreateSemaphore args;

    if (!ctx->dispatch_vkCreateSemaphore) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    vn_decode_vkCreateSemaphore_args_temp(ctx->decoder, &args);
    if (!args.device) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    if (!vn_cs_decoder_get_fatal(ctx->decoder))
        ctx->dispatch_vkCreateSemaphore(ctx, &args);

#ifdef DEBUG
    if (!vn_cs_decoder_get_fatal(ctx->decoder) && vn_dispatch_should_log_result(args.ret))
        vn_dispatch_debug_log(ctx, "vkCreateSemaphore returned %d", args.ret);
#endif

    if ((flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT) && !vn_cs_decoder_get_fatal(ctx->decoder))
        vn_encode_vkCreateSemaphore_reply(ctx->encoder, &args);

    vn_cs_decoder_reset_temp_pool(ctx->decoder);
}

static inline void vn_dispatch_vkDestroySemaphore(struct vn_dispatch_context *ctx, VkCommandFlagsEXT flags)
{
    struct vn_command_vkDestroySemaphore args;

    if (!ctx->dispatch_vkDestroySemaphore) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    vn_decode_vkDestroySemaphore_args_temp(ctx->decoder, &args);
    if (!args.device) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    if (!vn_cs_decoder_get_fatal(ctx->decoder))
        ctx->dispatch_vkDestroySemaphore(ctx, &args);

    if ((flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT) && !vn_cs_decoder_get_fatal(ctx->decoder))
        vn_encode_vkDestroySemaphore_reply(ctx->encoder, &args);

    vn_cs_decoder_reset_temp_pool(ctx->decoder);
}

static inline void vn_dispatch_vkGetSemaphoreCounterValue(struct vn_dispatch_context *ctx, VkCommandFlagsEXT flags)
{
    struct vn_command_vkGetSemaphoreCounterValue args;

    if (!ctx->dispatch_vkGetSemaphoreCounterValue) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    vn_decode_vkGetSemaphoreCounterValue_args_temp(ctx->decoder, &args);
    if (!args.device) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    if (!vn_cs_decoder_get_fatal(ctx->decoder))
        ctx->dispatch_vkGetSemaphoreCounterValue(ctx, &args);

#ifdef DEBUG
    if (!vn_cs_decoder_get_fatal(ctx->decoder) && vn_dispatch_should_log_result(args.ret))
        vn_dispatch_debug_log(ctx, "vkGetSemaphoreCounterValue returned %d", args.ret);
#endif

    if (flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT) {
        if (!vn_cs_decoder_get_fatal(ctx->decoder))
            vn_encode_vkGetSemaphoreCounterValue_reply(ctx->encoder, &args);
    } else if (args.ret == VK_ERROR_DEVICE_LOST) {
        vn_cs_decoder_set_fatal(ctx->decoder);
    }

    vn_cs_decoder_reset_temp_pool(ctx->decoder);
}

static inline void vn_dispatch_vkWaitSemaphores(struct vn_dispatch_context *ctx, VkCommandFlagsEXT flags)
{
    struct vn_command_vkWaitSemaphores args;

    if (!ctx->dispatch_vkWaitSemaphores) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    vn_decode_vkWaitSemaphores_args_temp(ctx->decoder, &args);
    if (!args.device) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    if (!vn_cs_decoder_get_fatal(ctx->decoder))
        ctx->dispatch_vkWaitSemaphores(ctx, &args);

#ifdef DEBUG
    if (!vn_cs_decoder_get_fatal(ctx->decoder) && vn_dispatch_should_log_result(args.ret))
        vn_dispatch_debug_log(ctx, "vkWaitSemaphores returned %d", args.ret);
#endif

    if (flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT) {
        if (!vn_cs_decoder_get_fatal(ctx->decoder))
            vn_encode_vkWaitSemaphores_reply(ctx->encoder, &args);
    } else if (args.ret == VK_ERROR_DEVICE_LOST) {
        vn_cs_decoder_set_fatal(ctx->decoder);
    }

    vn_cs_decoder_reset_temp_pool(ctx->decoder);
}

static inline void vn_dispatch_vkSignalSemaphore(struct vn_dispatch_context *ctx, VkCommandFlagsEXT flags)
{
    struct vn_command_vkSignalSemaphore args;

    if (!ctx->dispatch_vkSignalSemaphore) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    vn_decode_vkSignalSemaphore_args_temp(ctx->decoder, &args);
    if (!args.device) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    if (!vn_cs_decoder_get_fatal(ctx->decoder))
        ctx->dispatch_vkSignalSemaphore(ctx, &args);

#ifdef DEBUG
    if (!vn_cs_decoder_get_fatal(ctx->decoder) && vn_dispatch_should_log_result(args.ret))
        vn_dispatch_debug_log(ctx, "vkSignalSemaphore returned %d", args.ret);
#endif

    if ((flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT) && !vn_cs_decoder_get_fatal(ctx->decoder))
        vn_encode_vkSignalSemaphore_reply(ctx->encoder, &args);

    vn_cs_decoder_reset_temp_pool(ctx->decoder);
}

static inline void vn_dispatch_vkWaitSemaphoreResource100000MESA(struct vn_dispatch_context *ctx, VkCommandFlagsEXT flags)
{
    struct vn_command_vkWaitSemaphoreResource100000MESA args;

    if (!ctx->dispatch_vkWaitSemaphoreResource100000MESA) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    vn_decode_vkWaitSemaphoreResource100000MESA_args_temp(ctx->decoder, &args);
    if (!args.device) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    if (!vn_cs_decoder_get_fatal(ctx->decoder))
        ctx->dispatch_vkWaitSemaphoreResource100000MESA(ctx, &args);

    if ((flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT) && !vn_cs_decoder_get_fatal(ctx->decoder))
        vn_encode_vkWaitSemaphoreResource100000MESA_reply(ctx->encoder, &args);

    vn_cs_decoder_reset_temp_pool(ctx->decoder);
}

static inline void vn_dispatch_vkImportSemaphoreResource100000MESA(struct vn_dispatch_context *ctx, VkCommandFlagsEXT flags)
{
    struct vn_command_vkImportSemaphoreResource100000MESA args;

    if (!ctx->dispatch_vkImportSemaphoreResource100000MESA) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    vn_decode_vkImportSemaphoreResource100000MESA_args_temp(ctx->decoder, &args);
    if (!args.device) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    if (!vn_cs_decoder_get_fatal(ctx->decoder))
        ctx->dispatch_vkImportSemaphoreResource100000MESA(ctx, &args);

    if ((flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT) && !vn_cs_decoder_get_fatal(ctx->decoder))
        vn_encode_vkImportSemaphoreResource100000MESA_reply(ctx->encoder, &args);

    vn_cs_decoder_reset_temp_pool(ctx->decoder);
}

#pragma GCC diagnostic pop

#endif /* VN_PROTOCOL_RENDERER_SEMAPHORE_H */
