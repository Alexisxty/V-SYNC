"""
Extended vision inference that supports text questions.
扩展的视觉推理,支持文本问题输入。

This module extends Mini-Omni2's vision capabilities to accept text questions
in addition to audio and image inputs.
"""
import torch
from inference_vision import OmniVisionInference, load_clip_model, _image, _eoimage, _pad_a, _eoa, _pad_t, _eot, _input_a, _answer_a, _input_t, _answer_t
from inference import load_audio, text_vocabsize, padded_text_vocabsize, get_text_stream
from utils.snac_utils import layershift, reconscruct_snac, reconstruct_tensors, get_snac, generate_audio_data
from litgpt.generate.base import next_token_image_batch
from PIL import Image
from tqdm import tqdm


def get_input_ids_ImageQA_Text_Batch(mel, leng, text_question, whispermodel, text_tokenizer, device):
    """
    Create input IDs for Image + Audio + Text Question task.
    FIXED: Ensure input_ids length matches the positions model will write to.

    Args:
        mel: audio mel spectrogram
        leng: audio length
        text_question: text question string
        whispermodel: whisper model for audio encoding
        text_tokenizer: tokenizer for text
        device: cuda device

    Returns:
        audio_feature: encoded audio features (stacked)
        input_ids: formatted input IDs for the model
    """
    # Encode audio
    with torch.no_grad():
        mel = mel.unsqueeze(0).to(device)
        audio_feature = whispermodel.embed_audio(mel)[0][:leng]

    audio_len = audio_feature.size(0)

    # CRITICAL: Hard limit to 1500 frames (30 seconds) to avoid tensor size mismatch
    # The model's concat_feat has a fixed buffer: positions 52 to 1552 (1500 frames max)
    if audio_len > 1500:
        print(f"Warning: Audio length {audio_len} exceeds maximum 1500, truncating to 1500")
        audio_feature = audio_feature[:1500]
        audio_len = 1500

    # Tokenize text question
    text_tokens = text_tokenizer.encode(text_question).tolist()
    text_len = len(text_tokens)

    input_ids = []

    # First batch: Image + Audio + Text Question → Audio Answer
    # Total length = 52 (image) + audio_len+2 (audio) + text_len+2 (text) + 1 (answer)
    total_len = 52 + (audio_len + 2) + (text_len + 2) + 1

    input_ids_item = [[] for i in range(8)]
    for i in range(7):
        # Image tokens: 1 + 50 + 1 = 52
        input_ids_item[i] = [layershift(_image, i)] + [layershift(_pad_a, i)] * 50 + [layershift(_eoimage, i)]
        # Audio tokens: 1 + audio_len + 1
        input_ids_item[i] += [layershift(_input_a, i)] + [layershift(_pad_a, i)] * audio_len + [layershift(_eoa, i)]
        # Text question padding: text_len + 2
        input_ids_item[i] += [layershift(_pad_a, i)] * (text_len + 2)
        # Answer token
        input_ids_item[i] += [layershift(_answer_a, i)]

    # Text layer: MUST match the same total length
    input_ids_item[-1] = [_pad_t] * 52  # Image placeholders
    input_ids_item[-1] += [_pad_t] * (audio_len + 2)  # Audio placeholders
    input_ids_item[-1] += [_input_t] + text_tokens + [_eot]  # Text question
    input_ids_item[-1] += [_answer_t]  # Answer token

    input_ids_item = [torch.tensor(item) for item in input_ids_item]
    input_ids.append(input_ids_item)

    # Second batch: Image + Audio + Text Question → Text Answer
    input_ids_item = [[] for i in range(8)]
    for i in range(7):
        # Image tokens: 52
        input_ids_item[i] = [layershift(_image, i)] + [layershift(_pad_a, i)] * 50 + [layershift(_eoimage, i)]
        # Audio tokens: audio_len + 2
        input_ids_item[i] += [layershift(_input_a, i)] + [layershift(_pad_a, i)] * audio_len + [layershift(_eoa, i)]
        # Text question padding: text_len + 2
        input_ids_item[i] += [layershift(_pad_a, i)] * (text_len + 2)
        # Answer padding: 1
        input_ids_item[i] += [layershift(_pad_a, i)]

    # Text layer
    input_ids_item[-1] = [_pad_t] * 52  # Image placeholders
    input_ids_item[-1] += [_pad_t] * (audio_len + 2)  # Audio placeholders
    input_ids_item[-1] += [_input_t] + text_tokens + [_eot]  # Text question
    input_ids_item[-1] += [_answer_t]  # Answer token

    input_ids_item = [torch.tensor(item) for item in input_ids_item]
    input_ids.append(input_ids_item)

    # Stack inputs
    stacked_inputids = [[] for _ in range(8)]
    for i in range(2):
        for j in range(8):
            stacked_inputids[j].append(input_ids[i][j])
    stacked_inputids = [torch.stack(tensors) for tensors in stacked_inputids]

    return torch.stack([audio_feature, audio_feature]), stacked_inputids


class OmniVisionTextInference(OmniVisionInference):
    """
    Extended OmniVisionInference that supports text questions.
    支持文本问题的扩展视觉推理类。
    """

    @torch.inference_mode()
    def run_vision_with_text_question(
        self,
        audio_path,
        image_path,
        text_question,
        stream_stride=4,
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        top_p=1.0,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id=_pad_t,
        save_path=None,
        warm_up=False
    ):
        """
        Run inference with audio, image, and text question.

        Args:
            audio_path: path to audio file
            image_path: path to image file
            text_question: text question string
            ... (other parameters same as run_vision_AA_batch_stream)

        Yields:
            (audio_stream, text_stream) tuples
        """
        with self.fabric.init_tensor():
            self.model.set_kv_cache(batch_size=2)

        model = self.model

        # Load audio and image
        mel, leng = load_audio(audio_path)
        img = Image.open(image_path)

        # Get input IDs with text question
        audio_feature, input_ids = get_input_ids_ImageQA_Text_Batch(
            mel, leng, text_question,
            self.whispermodel, self.text_tokenizer, self.device
        )

        # Encode image
        ima = self.clippreprocess(img).unsqueeze(0).to(self.device)
        ima_feature = self.clipmodel.encode_image(ima).squeeze(0).to(self.device)
        ima_feature = torch.stack([ima_feature.clone(), ima_feature.clone()]).to(self.device)

        leng = [leng, leng]
        task = ['ImageQA_A', 'ImageQA_AT']

        T = input_ids[0].size(1)
        assert max_returned_tokens > T, f"max_returned_tokens {max_returned_tokens} should be greater than input length {T}"

        if model.max_seq_length < max_returned_tokens - 1:
            raise NotImplementedError(
                f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}"
            )

        list_output = [[] for i in range(8)]

        # First token generation
        tokens_A, token_T = next_token_image_batch(
            model,
            audio_feature.to(torch.float32).to(self.device),
            ima_feature.to(torch.float32).to(self.device),
            input_ids,
            whisper_lens=leng,
            task=task,
            input_pos=torch.arange(0, T, device=self.device),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        for i in range(7):
            list_output[i].append(tokens_A[i].tolist()[0])
        list_output[7].append(token_T.tolist()[0])

        text_end = False
        index = 1
        nums_generate = stream_stride
        begin_generate = False
        current_index = 0
        input_pos = torch.tensor([T], device=self.device)

        model_input_ids = [[] for i in range(8)]
        for i in range(7):
            tokens_A[i] = tokens_A[i].clone() + padded_text_vocabsize + i * 4160
            model_input_ids[i].append(tokens_A[i].clone().to(self.device).to(torch.int32))
            model_input_ids[i].append(torch.tensor([layershift(4097, i)], device=self.device))
            model_input_ids[i] = torch.stack(model_input_ids[i])

        model_input_ids[-1].append(token_T.clone().to(torch.int32))
        model_input_ids[-1].append(token_T.clone().to(torch.int32))
        model_input_ids[-1] = torch.stack(model_input_ids[-1])

        text_index = 0
        is_text_end = False

        # Generate remaining tokens
        for _ in tqdm(range(2, max_returned_tokens - T + 1), disable=warm_up):
            tokens_A, token_T = next_token_image_batch(
                model, None, None,
                input_ids=model_input_ids,
                whisper_lens=None,
                task=None,
                input_pos=input_pos,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            if text_end:
                token_T = torch.tensor([_pad_t], device=self.device)

            if tokens_A[-1] == eos_id_a:
                break
            if token_T == eos_id_t:
                text_end = True

            for i in range(7):
                list_output[i].append(tokens_A[i].tolist()[0])
            list_output[7].append(token_T.tolist()[0])

            if index == 7:
                begin_generate = True

            if begin_generate:
                current_index += 1
                if current_index == nums_generate:
                    current_index = 0
                    snac = get_snac(list_output, index, nums_generate)
                    audio_stream = generate_audio_data(snac, self.snacmodel, self.device)

                    if is_text_end:
                        text_stream = ""
                    else:
                        text_stream, text_index, is_text_end = get_text_stream(list_output, text_index, self.text_tokenizer)

                    yield (audio_stream, text_stream)

                    if warm_up:
                        break

            input_pos = input_pos.add_(1)
            model_input_ids = [[] for i in range(8)]
            for i in range(7):
                tokens_A[i] = tokens_A[i].clone() + padded_text_vocabsize + i * 4160
                model_input_ids[i].append(tokens_A[i].clone().to(self.device).to(torch.int32))
                model_input_ids[i].append(torch.tensor([layershift(4097, i)], device=self.device))
                model_input_ids[i] = torch.stack(model_input_ids[i])

            model_input_ids[-1].append(token_T.clone().to(torch.int32))
            model_input_ids[-1].append(token_T.clone().to(torch.int32))
            model_input_ids[-1] = torch.stack(model_input_ids[-1])

            index += 1

        # Extract final text
        text_tokens = list_output[-1]
        if text_vocabsize in text_tokens:
            text_tokens = text_tokens[:text_tokens.index(text_vocabsize)]
        res_text = self.text_tokenizer.decode(torch.tensor(text_tokens))

        if not warm_up:
            print(f"text output: {res_text}")

        # Save audio if requested
        if save_path is not None:
            import soundfile as sf
            audiolist = reconscruct_snac(list_output)
            audio = reconstruct_tensors(audiolist)
            with torch.inference_mode():
                audio_hat = self.snacmodel.decode(audio)
                sf.write(save_path, audio_hat.squeeze().cpu().numpy(), 24000)

        model.clear_kv_cache()
