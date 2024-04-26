/* This file is generated by venus-protocol.  See vn_protocol_renderer.h. */

/*
 * Copyright 2020 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef VN_PROTOCOL_RENDERER_DESCRIPTOR_SET_H
#define VN_PROTOCOL_RENDERER_DESCRIPTOR_SET_H

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
 *   vkUpdateDescriptorSetWithTemplate
 */

/* struct VkDescriptorSetVariableDescriptorCountAllocateInfo chain */

static inline void *
vn_decode_VkDescriptorSetVariableDescriptorCountAllocateInfo_pnext_temp(struct vn_cs_decoder *dec)
{
    /* no known/supported struct */
    if (vn_decode_simple_pointer(dec))
        vn_cs_decoder_set_fatal(dec);
    return NULL;
}

static inline void
vn_decode_VkDescriptorSetVariableDescriptorCountAllocateInfo_self_temp(struct vn_cs_decoder *dec, VkDescriptorSetVariableDescriptorCountAllocateInfo *val)
{
    /* skip val->{sType,pNext} */
    vn_decode_uint32_t(dec, &val->descriptorSetCount);
    if (vn_peek_array_size(dec)) {
        const size_t array_size = vn_decode_array_size(dec, val->descriptorSetCount);
        val->pDescriptorCounts = vn_cs_decoder_alloc_temp_array(dec, sizeof(*val->pDescriptorCounts), array_size);
        if (!val->pDescriptorCounts) return;
        vn_decode_uint32_t_array(dec, (uint32_t *)val->pDescriptorCounts, array_size);
    } else {
        vn_decode_array_size(dec, val->descriptorSetCount);
        val->pDescriptorCounts = NULL;
    }
}

static inline void
vn_decode_VkDescriptorSetVariableDescriptorCountAllocateInfo_temp(struct vn_cs_decoder *dec, VkDescriptorSetVariableDescriptorCountAllocateInfo *val)
{
    VkStructureType stype;
    vn_decode_VkStructureType(dec, &stype);
    if (stype != VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO)
        vn_cs_decoder_set_fatal(dec);

    val->sType = stype;
    val->pNext = vn_decode_VkDescriptorSetVariableDescriptorCountAllocateInfo_pnext_temp(dec);
    vn_decode_VkDescriptorSetVariableDescriptorCountAllocateInfo_self_temp(dec, val);
}

static inline void
vn_replace_VkDescriptorSetVariableDescriptorCountAllocateInfo_handle_self(VkDescriptorSetVariableDescriptorCountAllocateInfo *val)
{
    /* skip val->sType */
    /* skip val->pNext */
    /* skip val->descriptorSetCount */
    /* skip val->pDescriptorCounts */
}

static inline void
vn_replace_VkDescriptorSetVariableDescriptorCountAllocateInfo_handle(VkDescriptorSetVariableDescriptorCountAllocateInfo *val)
{
    struct VkBaseOutStructure *pnext = (struct VkBaseOutStructure *)val;

    do {
        switch ((int32_t)pnext->sType) {
        case VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO:
            vn_replace_VkDescriptorSetVariableDescriptorCountAllocateInfo_handle_self((VkDescriptorSetVariableDescriptorCountAllocateInfo *)pnext);
            break;
        default:
            /* ignore unknown/unsupported struct */
            break;
        }
        pnext = pnext->pNext;
    } while (pnext);
}

/* struct VkDescriptorSetAllocateInfo chain */

static inline void *
vn_decode_VkDescriptorSetAllocateInfo_pnext_temp(struct vn_cs_decoder *dec)
{
    VkBaseOutStructure *pnext;
    VkStructureType stype;

    if (!vn_decode_simple_pointer(dec))
        return NULL;

    vn_decode_VkStructureType(dec, &stype);
    switch ((int32_t)stype) {
    case VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO:
        pnext = vn_cs_decoder_alloc_temp(dec, sizeof(VkDescriptorSetVariableDescriptorCountAllocateInfo));
        if (pnext) {
            pnext->sType = stype;
            pnext->pNext = vn_decode_VkDescriptorSetAllocateInfo_pnext_temp(dec);
            vn_decode_VkDescriptorSetVariableDescriptorCountAllocateInfo_self_temp(dec, (VkDescriptorSetVariableDescriptorCountAllocateInfo *)pnext);
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
vn_decode_VkDescriptorSetAllocateInfo_self_temp(struct vn_cs_decoder *dec, VkDescriptorSetAllocateInfo *val)
{
    /* skip val->{sType,pNext} */
    vn_decode_VkDescriptorPool_lookup(dec, &val->descriptorPool);
    vn_decode_uint32_t(dec, &val->descriptorSetCount);
    if (vn_peek_array_size(dec)) {
        const uint32_t iter_count = vn_decode_array_size(dec, val->descriptorSetCount);
        val->pSetLayouts = vn_cs_decoder_alloc_temp_array(dec, sizeof(*val->pSetLayouts), iter_count);
        if (!val->pSetLayouts) return;
        for (uint32_t i = 0; i < iter_count; i++)
            vn_decode_VkDescriptorSetLayout_lookup(dec, &((VkDescriptorSetLayout *)val->pSetLayouts)[i]);
    } else {
        vn_decode_array_size(dec, val->descriptorSetCount);
        val->pSetLayouts = NULL;
    }
}

static inline void
vn_decode_VkDescriptorSetAllocateInfo_temp(struct vn_cs_decoder *dec, VkDescriptorSetAllocateInfo *val)
{
    VkStructureType stype;
    vn_decode_VkStructureType(dec, &stype);
    if (stype != VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
        vn_cs_decoder_set_fatal(dec);

    val->sType = stype;
    val->pNext = vn_decode_VkDescriptorSetAllocateInfo_pnext_temp(dec);
    vn_decode_VkDescriptorSetAllocateInfo_self_temp(dec, val);
}

static inline void
vn_replace_VkDescriptorSetAllocateInfo_handle_self(VkDescriptorSetAllocateInfo *val)
{
    /* skip val->sType */
    /* skip val->pNext */
    vn_replace_VkDescriptorPool_handle(&val->descriptorPool);
    /* skip val->descriptorSetCount */
    if (val->pSetLayouts) {
       for (uint32_t i = 0; i < val->descriptorSetCount; i++)
            vn_replace_VkDescriptorSetLayout_handle(&((VkDescriptorSetLayout *)val->pSetLayouts)[i]);
    }
}

static inline void
vn_replace_VkDescriptorSetAllocateInfo_handle(VkDescriptorSetAllocateInfo *val)
{
    struct VkBaseOutStructure *pnext = (struct VkBaseOutStructure *)val;

    do {
        switch ((int32_t)pnext->sType) {
        case VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO:
            vn_replace_VkDescriptorSetAllocateInfo_handle_self((VkDescriptorSetAllocateInfo *)pnext);
            break;
        case VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO:
            vn_replace_VkDescriptorSetVariableDescriptorCountAllocateInfo_handle_self((VkDescriptorSetVariableDescriptorCountAllocateInfo *)pnext);
            break;
        default:
            /* ignore unknown/unsupported struct */
            break;
        }
        pnext = pnext->pNext;
    } while (pnext);
}

/* struct VkCopyDescriptorSet chain */

static inline void *
vn_decode_VkCopyDescriptorSet_pnext_temp(struct vn_cs_decoder *dec)
{
    /* no known/supported struct */
    if (vn_decode_simple_pointer(dec))
        vn_cs_decoder_set_fatal(dec);
    return NULL;
}

static inline void
vn_decode_VkCopyDescriptorSet_self_temp(struct vn_cs_decoder *dec, VkCopyDescriptorSet *val)
{
    /* skip val->{sType,pNext} */
    vn_decode_VkDescriptorSet_lookup(dec, &val->srcSet);
    vn_decode_uint32_t(dec, &val->srcBinding);
    vn_decode_uint32_t(dec, &val->srcArrayElement);
    vn_decode_VkDescriptorSet_lookup(dec, &val->dstSet);
    vn_decode_uint32_t(dec, &val->dstBinding);
    vn_decode_uint32_t(dec, &val->dstArrayElement);
    vn_decode_uint32_t(dec, &val->descriptorCount);
}

static inline void
vn_decode_VkCopyDescriptorSet_temp(struct vn_cs_decoder *dec, VkCopyDescriptorSet *val)
{
    VkStructureType stype;
    vn_decode_VkStructureType(dec, &stype);
    if (stype != VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET)
        vn_cs_decoder_set_fatal(dec);

    val->sType = stype;
    val->pNext = vn_decode_VkCopyDescriptorSet_pnext_temp(dec);
    vn_decode_VkCopyDescriptorSet_self_temp(dec, val);
}

static inline void
vn_replace_VkCopyDescriptorSet_handle_self(VkCopyDescriptorSet *val)
{
    /* skip val->sType */
    /* skip val->pNext */
    vn_replace_VkDescriptorSet_handle(&val->srcSet);
    /* skip val->srcBinding */
    /* skip val->srcArrayElement */
    vn_replace_VkDescriptorSet_handle(&val->dstSet);
    /* skip val->dstBinding */
    /* skip val->dstArrayElement */
    /* skip val->descriptorCount */
}

static inline void
vn_replace_VkCopyDescriptorSet_handle(VkCopyDescriptorSet *val)
{
    struct VkBaseOutStructure *pnext = (struct VkBaseOutStructure *)val;

    do {
        switch ((int32_t)pnext->sType) {
        case VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET:
            vn_replace_VkCopyDescriptorSet_handle_self((VkCopyDescriptorSet *)pnext);
            break;
        default:
            /* ignore unknown/unsupported struct */
            break;
        }
        pnext = pnext->pNext;
    } while (pnext);
}

static inline void vn_decode_vkAllocateDescriptorSets_args_temp(struct vn_cs_decoder *dec, struct vn_command_vkAllocateDescriptorSets *args)
{
    vn_decode_VkDevice_lookup(dec, &args->device);
    if (vn_decode_simple_pointer(dec)) {
        args->pAllocateInfo = vn_cs_decoder_alloc_temp(dec, sizeof(*args->pAllocateInfo));
        if (!args->pAllocateInfo) return;
        vn_decode_VkDescriptorSetAllocateInfo_temp(dec, (VkDescriptorSetAllocateInfo *)args->pAllocateInfo);
    } else {
        args->pAllocateInfo = NULL;
        vn_cs_decoder_set_fatal(dec);
    }
    if (vn_peek_array_size(dec)) {
        const uint32_t iter_count = vn_decode_array_size(dec, (args->pAllocateInfo ? args->pAllocateInfo->descriptorSetCount : 0));
        args->pDescriptorSets = vn_cs_decoder_alloc_temp_array(dec, sizeof(*args->pDescriptorSets), iter_count);
        if (!args->pDescriptorSets) return;
        for (uint32_t i = 0; i < iter_count; i++)
            vn_decode_VkDescriptorSet(dec, &args->pDescriptorSets[i]);
    } else {
        vn_decode_array_size(dec, (args->pAllocateInfo ? args->pAllocateInfo->descriptorSetCount : 0));
        args->pDescriptorSets = NULL;
    }
}

static inline void vn_replace_vkAllocateDescriptorSets_args_handle(struct vn_command_vkAllocateDescriptorSets *args)
{
    vn_replace_VkDevice_handle(&args->device);
    if (args->pAllocateInfo)
        vn_replace_VkDescriptorSetAllocateInfo_handle((VkDescriptorSetAllocateInfo *)args->pAllocateInfo);
    /* skip args->pDescriptorSets */
}

static inline void vn_encode_vkAllocateDescriptorSets_reply(struct vn_cs_encoder *enc, const struct vn_command_vkAllocateDescriptorSets *args)
{
    vn_encode_VkCommandTypeEXT(enc, &(VkCommandTypeEXT){VK_COMMAND_TYPE_vkAllocateDescriptorSets_EXT});

    vn_encode_VkResult(enc, &args->ret);
    /* skip args->device */
    /* skip args->pAllocateInfo */
    if (args->pDescriptorSets) {
        vn_encode_array_size(enc, (args->pAllocateInfo ? args->pAllocateInfo->descriptorSetCount : 0));
        for (uint32_t i = 0; i < (args->pAllocateInfo ? args->pAllocateInfo->descriptorSetCount : 0); i++)
            vn_encode_VkDescriptorSet(enc, &args->pDescriptorSets[i]);
    } else {
        vn_encode_array_size(enc, 0);
    }
}

static inline void vn_decode_vkFreeDescriptorSets_args_temp(struct vn_cs_decoder *dec, struct vn_command_vkFreeDescriptorSets *args)
{
    vn_decode_VkDevice_lookup(dec, &args->device);
    vn_decode_VkDescriptorPool_lookup(dec, &args->descriptorPool);
    vn_decode_uint32_t(dec, &args->descriptorSetCount);
    if (vn_peek_array_size(dec)) {
        const uint32_t iter_count = vn_decode_array_size(dec, args->descriptorSetCount);
        args->pDescriptorSets = vn_cs_decoder_alloc_temp_array(dec, sizeof(*args->pDescriptorSets), iter_count);
        if (!args->pDescriptorSets) return;
        for (uint32_t i = 0; i < iter_count; i++)
            vn_decode_VkDescriptorSet_lookup(dec, &((VkDescriptorSet *)args->pDescriptorSets)[i]);
    } else {
        vn_decode_array_size_unchecked(dec);
        args->pDescriptorSets = NULL;
    }
}

static inline void vn_replace_vkFreeDescriptorSets_args_handle(struct vn_command_vkFreeDescriptorSets *args)
{
    vn_replace_VkDevice_handle(&args->device);
    vn_replace_VkDescriptorPool_handle(&args->descriptorPool);
    /* skip args->descriptorSetCount */
    if (args->pDescriptorSets) {
       for (uint32_t i = 0; i < args->descriptorSetCount; i++)
            vn_replace_VkDescriptorSet_handle(&((VkDescriptorSet *)args->pDescriptorSets)[i]);
    }
}

static inline void vn_encode_vkFreeDescriptorSets_reply(struct vn_cs_encoder *enc, const struct vn_command_vkFreeDescriptorSets *args)
{
    vn_encode_VkCommandTypeEXT(enc, &(VkCommandTypeEXT){VK_COMMAND_TYPE_vkFreeDescriptorSets_EXT});

    vn_encode_VkResult(enc, &args->ret);
    /* skip args->device */
    /* skip args->descriptorPool */
    /* skip args->descriptorSetCount */
    /* skip args->pDescriptorSets */
}

static inline void vn_decode_vkUpdateDescriptorSets_args_temp(struct vn_cs_decoder *dec, struct vn_command_vkUpdateDescriptorSets *args)
{
    vn_decode_VkDevice_lookup(dec, &args->device);
    vn_decode_uint32_t(dec, &args->descriptorWriteCount);
    if (vn_peek_array_size(dec)) {
        const uint32_t iter_count = vn_decode_array_size(dec, args->descriptorWriteCount);
        args->pDescriptorWrites = vn_cs_decoder_alloc_temp_array(dec, sizeof(*args->pDescriptorWrites), iter_count);
        if (!args->pDescriptorWrites) return;
        for (uint32_t i = 0; i < iter_count; i++)
            vn_decode_VkWriteDescriptorSet_temp(dec, &((VkWriteDescriptorSet *)args->pDescriptorWrites)[i]);
    } else {
        vn_decode_array_size(dec, args->descriptorWriteCount);
        args->pDescriptorWrites = NULL;
    }
    vn_decode_uint32_t(dec, &args->descriptorCopyCount);
    if (vn_peek_array_size(dec)) {
        const uint32_t iter_count = vn_decode_array_size(dec, args->descriptorCopyCount);
        args->pDescriptorCopies = vn_cs_decoder_alloc_temp_array(dec, sizeof(*args->pDescriptorCopies), iter_count);
        if (!args->pDescriptorCopies) return;
        for (uint32_t i = 0; i < iter_count; i++)
            vn_decode_VkCopyDescriptorSet_temp(dec, &((VkCopyDescriptorSet *)args->pDescriptorCopies)[i]);
    } else {
        vn_decode_array_size(dec, args->descriptorCopyCount);
        args->pDescriptorCopies = NULL;
    }
}

static inline void vn_replace_vkUpdateDescriptorSets_args_handle(struct vn_command_vkUpdateDescriptorSets *args)
{
    vn_replace_VkDevice_handle(&args->device);
    /* skip args->descriptorWriteCount */
    if (args->pDescriptorWrites) {
       for (uint32_t i = 0; i < args->descriptorWriteCount; i++)
            vn_replace_VkWriteDescriptorSet_handle(&((VkWriteDescriptorSet *)args->pDescriptorWrites)[i]);
    }
    /* skip args->descriptorCopyCount */
    if (args->pDescriptorCopies) {
       for (uint32_t i = 0; i < args->descriptorCopyCount; i++)
            vn_replace_VkCopyDescriptorSet_handle(&((VkCopyDescriptorSet *)args->pDescriptorCopies)[i]);
    }
}

static inline void vn_encode_vkUpdateDescriptorSets_reply(struct vn_cs_encoder *enc, const struct vn_command_vkUpdateDescriptorSets *args)
{
    vn_encode_VkCommandTypeEXT(enc, &(VkCommandTypeEXT){VK_COMMAND_TYPE_vkUpdateDescriptorSets_EXT});

    /* skip args->device */
    /* skip args->descriptorWriteCount */
    /* skip args->pDescriptorWrites */
    /* skip args->descriptorCopyCount */
    /* skip args->pDescriptorCopies */
}

static inline void vn_dispatch_vkAllocateDescriptorSets(struct vn_dispatch_context *ctx, VkCommandFlagsEXT flags)
{
    struct vn_command_vkAllocateDescriptorSets args;

    if (!ctx->dispatch_vkAllocateDescriptorSets) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    vn_decode_vkAllocateDescriptorSets_args_temp(ctx->decoder, &args);
    if (!args.device) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    if (!vn_cs_decoder_get_fatal(ctx->decoder))
        ctx->dispatch_vkAllocateDescriptorSets(ctx, &args);

#ifdef DEBUG
    if (!vn_cs_decoder_get_fatal(ctx->decoder) && vn_dispatch_should_log_result(args.ret))
        vn_dispatch_debug_log(ctx, "vkAllocateDescriptorSets returned %d", args.ret);
#endif

    if ((flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT) && !vn_cs_decoder_get_fatal(ctx->decoder)) {
        if (vn_cs_encoder_acquire(ctx->encoder)) {
            vn_encode_vkAllocateDescriptorSets_reply(ctx->encoder, &args);
            vn_cs_encoder_release(ctx->encoder);
        }
    }

    vn_cs_decoder_reset_temp_pool(ctx->decoder);
}

static inline void vn_dispatch_vkFreeDescriptorSets(struct vn_dispatch_context *ctx, VkCommandFlagsEXT flags)
{
    struct vn_command_vkFreeDescriptorSets args;

    if (!ctx->dispatch_vkFreeDescriptorSets) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    vn_decode_vkFreeDescriptorSets_args_temp(ctx->decoder, &args);
    if (!args.device) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    if (!vn_cs_decoder_get_fatal(ctx->decoder))
        ctx->dispatch_vkFreeDescriptorSets(ctx, &args);

#ifdef DEBUG
    if (!vn_cs_decoder_get_fatal(ctx->decoder) && vn_dispatch_should_log_result(args.ret))
        vn_dispatch_debug_log(ctx, "vkFreeDescriptorSets returned %d", args.ret);
#endif

    if ((flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT) && !vn_cs_decoder_get_fatal(ctx->decoder)) {
        if (vn_cs_encoder_acquire(ctx->encoder)) {
            vn_encode_vkFreeDescriptorSets_reply(ctx->encoder, &args);
            vn_cs_encoder_release(ctx->encoder);
        }
    }

    vn_cs_decoder_reset_temp_pool(ctx->decoder);
}

static inline void vn_dispatch_vkUpdateDescriptorSets(struct vn_dispatch_context *ctx, VkCommandFlagsEXT flags)
{
    struct vn_command_vkUpdateDescriptorSets args;

    if (!ctx->dispatch_vkUpdateDescriptorSets) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    vn_decode_vkUpdateDescriptorSets_args_temp(ctx->decoder, &args);
    if (!args.device) {
        vn_cs_decoder_set_fatal(ctx->decoder);
        return;
    }

    if (!vn_cs_decoder_get_fatal(ctx->decoder))
        ctx->dispatch_vkUpdateDescriptorSets(ctx, &args);

    if ((flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT) && !vn_cs_decoder_get_fatal(ctx->decoder)) {
        if (vn_cs_encoder_acquire(ctx->encoder)) {
            vn_encode_vkUpdateDescriptorSets_reply(ctx->encoder, &args);
            vn_cs_encoder_release(ctx->encoder);
        }
    }

    vn_cs_decoder_reset_temp_pool(ctx->decoder);
}

#pragma GCC diagnostic pop

#endif /* VN_PROTOCOL_RENDERER_DESCRIPTOR_SET_H */
