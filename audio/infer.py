import argparse
import torch
import os
import json
# from tqdm import tqdm
import whisper

from omni_speech.constants import SPEECH_TOKEN_INDEX #, DEFAULT_SPEECH_TOKEN
from omni_speech.conversation import conv_templates # , SeparatorStyle
from omni_speech.model.builder import load_pretrained_model
from omni_speech.utils import disable_torch_init
from omni_speech.datasets.preprocess import tokenizer_speech_token
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math
import random


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, args, questions, tokenizer, model_config, input_type, mel_size, conv_mode):
        self.questions = questions
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.input_type = input_type
        self.mel_size = mel_size
        self.conv_mode = conv_mode
        ## in the transcription pool, we will store all the transcriptions in the dataset, so all the item["actual_prompts"]
        self.transcription_pool = [item["actual_prompt"] for item in questions]
        self.args = args

    def __getitem__(self, index):
        item = self.questions[index]
        speech_file = item["speech"]
        qs = item["conversations"][0]["value"]
        ## add the actual prompt to qs
        if self.args.measure_cosine_similarity_aligned:
            ## in this case we will use the actual prompt as the qs
            qs = qs + " " + item["actual_prompt"]
        elif self.args.measure_cosine_similarity_unaligned:
            ## in this case we will randomly choose a transcription from the pool
            qs = qs + " " + random.choice(self.transcription_pool)
            # print(qs, "SEEe", type(qs), "see actual prompt", item["actual_prompt"], len(self.transcription_pool))

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # assert os.path.exists(speech_file), f"File {speech_file} does not exist"
        
        speech = whisper.load_audio(speech_file)
        if self.input_type == "raw":
            speech = torch.from_numpy(speech)
            if self.model_config.speech_normalize:
                speech = torch.nn.functional.layer_norm(speech, speech.shape)
        elif self.input_type == "mel":
            speech = whisper.pad_or_trim(speech)
            speech = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0)

        input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt')

        return input_ids, speech, torch.LongTensor([speech.shape[0]])

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, speech_tensors, speech_lengths = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    speech_tensors = torch.stack(speech_tensors, dim=0)
    speech_lengths = torch.stack(speech_lengths, dim=0)
    return input_ids, speech_tensors, speech_lengths


def ctc_postprocess(tokens, blank):
    _toks = tokens.squeeze(0).tolist()
    deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
    hyp = [v for v in deduplicated_toks if v != blank]
    hyp = " ".join(list(map(str, hyp)))
    return hyp

# DataLoader
def create_data_loader(args, questions, tokenizer, model_config, input_type, mel_size, conv_mode, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(args, questions, tokenizer, model_config, input_type, mel_size, conv_mode)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def split_speech_text_embeddings(
    hidden_states,          # tuple(L+1) – each tensor (1, S_exp, d)
    speech_token_index: int,
    text_token_length: int,
):
    """
    Returns
    -------
    speech_vecs : list[Tensor]  (L,)  each (1, d)  – mean of speech span
    text_vecs   : list[Tensor]  (L,)  each (1, d)  – mean of text span
    cos_curve   : list[float]   (L,)  cosine(speech_vec, text_vec)
    """
    total_len = hidden_states[0].size(1)          # S_exp
    # slices [start : end] with end exclusive
    speech_slice = slice(speech_token_index, total_len - text_token_length)
    text_slice   = slice(total_len - text_token_length, total_len)

    speech_vecs, text_vecs, cos_curve = [], [], []
    for h in hidden_states[1:]:                   # skip embedding layer 0
        sp = h[:, speech_slice, :].mean(1)        # (1, d)
        tx = h[:, text_slice  , :].mean(1)        # (1, d)
        # speech_vecs.append(sp)
        # text_vecs.append(tx)
        cos_curve.append(F.cosine_similarity(sp, tx, dim=-1).item())

    return cos_curve


def jls_extract_def(args, data_loader, questions, answers_file, model, tokenizer):
    # ans_file = open(answers_file, "w")
    similarity_between_audio_and_transcription = []
    all_embeddings = []
    
    # for (input_ids, speech_tensor, speech_length), item in tqdm(zip(data_loader, questions), total=len(questions)):
    for (input_ids, speech_tensor, speech_length), item in zip(data_loader, questions):
        idx = item["id"]
        actual_prompt = item["actual_prompt"]
        try:
            answer = item["conversations"][1]["value"]
        except:
            answer = None
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        speech_tensor = speech_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
        speech_length = speech_length.to(device='cuda', non_blocking=True)
    
        with torch.inference_mode():
            if args.s2s:
                outputs = model.generate(
                    input_ids,
                    speech=speech_tensor,
                    speech_lengths=speech_length,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=128004,
                    streaming_unit_gen=False,
                )
                output_ids, output_units = outputs
    
            elif args.measure_cosine_similarity_aligned or args.measure_cosine_similarity_unaligned or args.compute_image_and_question_embeddings:
                out = model(
                    input_ids=input_ids,
                    speech=speech_tensor,
                    speech_lengths=speech_length,
                    output_hidden_states=True,
                    use_cache=False,          # we’re not decoding
                )
                if args.compute_image_and_question_embeddings:
                    ## here only return the hidden states at all layers. # return the hidden states  ## average the embeddings over the sequence length
                    embeddings = torch.stack(out.hidden_states).squeeze(1).mean(1)
                    print(f"Shape of the embeddings: {embeddings.shape}")
                    all_embeddings.append(embeddings)
                else:
                    ## find the index in input_ids where the value is -200, that is where speech tokens were inserted. 
                    speech_token_index = (input_ids == SPEECH_TOKEN_INDEX).nonzero(as_tuple=False)[0, 1].item()
                    ## text tokens are the things after this this index to the last token in the input_ids
                    text_token_length = input_ids.shape[1] - speech_token_index - 1
                    ## text embeddings are the last text_token_length embeddings in hidden_states at each layer
                    ## speech embeddings are the embeddings from index speech_token_index until the last text_token_length
                    cosine_sims = split_speech_text_embeddings(out.hidden_states, speech_token_index, text_token_length)
                    similarity_between_audio_and_transcription.append(cosine_sims)
                    print(len(similarity_between_audio_and_transcription))
    
            else:
                outputs = model.generate(
                    input_ids,
                    speech=speech_tensor,
                    speech_lengths=speech_length,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=128004,
                )
                output_ids = outputs
    
    
        if args.s2s:
            output_units = ctc_postprocess(output_units, blank=model.config.unit_vocab_size)
            print(f"H-{idx}\t{outputs}")
            print(f"T-{idx}\t{answer}")
            print(f"U-{idx}\t{output_units}")
            ans_file.write(json.dumps({"question_id": idx, "prediction": outputs, "prediction_units": output_units, "answer": answer}) + "\n")
        elif args.measure_cosine_similarity_aligned or args.measure_cosine_similarity_unaligned or args.compute_image_and_question_embeddings:
            pass
        else:
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(f"H-{idx}\t{outputs}")
            print(f"T-{idx}\t{answer}")
            ans_file.write(json.dumps({"question_id": idx, "actual_prompt": actual_prompt, "prediction": outputs, "answer": answer}) + "\n")
        # ans_file.flush()
    
    if args.measure_cosine_similarity_aligned:
        ## here we avg. the cosine similarities over the datapoints and make a plot of the cosine similarity (y-axis) vs. layers (x-axis)
        similarity_between_audio_and_transcription = torch.tensor(similarity_between_audio_and_transcription)
        assert similarity_between_audio_and_transcription.shape[0] == len(questions)
        similarity_between_audio_and_transcription = similarity_between_audio_and_transcription.mean(0)
        print(similarity_between_audio_and_transcription.shape, "final shape")
        ## save this as pt file
        torch.save(similarity_between_audio_and_transcription, f"cosine_similarity_aligned_{args.dataset}.pt")
        return None
    
    elif args.measure_cosine_similarity_unaligned:
        ## in this case just return the similarity_between_audio_and_transcription, this will be processed later. 
        similarity_between_audio_and_transcription = torch.tensor(similarity_between_audio_and_transcription)
        assert similarity_between_audio_and_transcription.shape[0] == len(questions)
        return similarity_between_audio_and_transcription
    
    elif args.compute_image_and_question_embeddings:
        all_embeddings = torch.stack(all_embeddings, dim=0)
        print(all_embeddings.shape, "final shape of all embeddings")
        ## save it
        torch.save(all_embeddings, f"full_dataset_embeddings_{args.dataset}.pt")
    # ans_file.close()


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    
    ## make a small dummy class model and make its config as None -- this is to avoid loading the model weights
    class DummyModel:
        def __init__(self):
            self.config = None
    model = DummyModel()
    tokenizer = None
    
    ## check if there is harmful in args.dataset, if there is then assert args.question_file should also have harmful. Same for benign
    if 'harmful' in args.dataset:
        assert 'harmful' in args.question_file, f"args.dataset: {args.dataset} and args.question_file: {args.question_file} are not consistent"
    elif 'benign' in args.dataset:
        assert 'benign' in args.question_file, f"args.dataset: {args.dataset} and args.question_file: {args.question_file} are not consistent"
    
    tokenizer, model, context_len = load_pretrained_model(model_path, args.model_base, is_lora=args.is_lora, s2s=args.s2s)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    ## take the first 5 questions:
    # questions = questions[:5]
    print(len(questions), "questions")
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    if args.measure_cosine_similarity_aligned or args.compute_image_and_question_embeddings:
        data_loader = create_data_loader(args, questions, tokenizer, model.config, args.input_type, args.mel_size, args.conv_mode)
        jls_extract_def(args, data_loader, questions, answers_file, model, tokenizer)
    elif args.measure_cosine_similarity_unaligned:
        similarities_over_shuffles = []
        for shuffle in range(10):
            data_loader = create_data_loader(args, questions, tokenizer, model.config, args.input_type, args.mel_size, args.conv_mode)
            similarity_between_audio_and_transcription = jls_extract_def(args, data_loader, questions, answers_file, model, tokenizer)
            print(similarity_between_audio_and_transcription.shape, "similarity shape")
            similarities_over_shuffles.append(similarity_between_audio_and_transcription)
        ## average the similarities over the shuffles
        similarities_over_shuffles = torch.stack(similarities_over_shuffles, dim=0)
        print(similarities_over_shuffles.shape, "similarity shape after stacking")
        similarities_over_shuffles = similarities_over_shuffles.mean(0)
        print(similarities_over_shuffles.shape, "similarity shape after averaging over shuffles")
        ## now average over the questions
        similarities_over_shuffles = similarities_over_shuffles.mean(0)
        print(similarities_over_shuffles.shape, "similarity shape after averaging over questions")
        ## save this as pt file
        torch.save(similarities_over_shuffles, f"cosine_similarity_unaligned_{args.dataset}.pt")
    else:
        raise NotImplementedError("Please choose either measure_cosine_similarity_aligned or measure_cosine_similarity_unaligned")
    

def plot_cosine_similarity(args):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    similarity_aligned = torch.load(f"cosine_similarity_aligned_{args.dataset}.pt")
    similarity_unaligned = torch.load(f"cosine_similarity_unaligned_{args.dataset}.pt")
    assert similarity_aligned.shape == similarity_unaligned.shape
    difference_aligned_and_unaligned = similarity_aligned - similarity_unaligned
    ## only plot the difference between aligned and unaligned
    plt.plot(difference_aligned_and_unaligned, label='Audios and Transcriptions', color='blue', marker='o', markersize=4)
    ## print the layer with the max difference
    print(f"Layer with max differences: {difference_aligned_and_unaligned.topk(5)}")
    plt.xlabel('Layer', fontsize=16)
    plt.ylabel('Universality Score (U-Score)', fontsize=14)
    plt.title('Layers with the highest U-Score', fontsize=16)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid()
    plt.ylim(-0.01, 0.12)
    plt.tight_layout()
    plt.savefig(f'cosine_similarity_{args.dataset}.pdf', dpi=300)
    print('Plot saved as cosine_similarity.pdf')


def train_classifier_on_embeddings(args):
    ## load the embeddings of two datasets, args.dataset and of mm_vet2
    best_layer = 21
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.use_each_dataset_experimental_design:
        
        if args.dataset == "advbench":
            harmful_image_question_embeddings = torch.load('full_dataset_embeddings_advbench.pt', weights_only=True)
        elif args.dataset == "toxigen":
            harmful_image_question_embeddings = torch.load("full_dataset_embeddings_toxigen_harmful.pt", weights_only=True)
            benign_image_question_embeddings = torch.load("full_dataset_embeddings_toxigen_benign.pt", weights_only=True)
        elif args.dataset == "aegis_safety":
            harmful_image_question_embeddings = torch.load("full_dataset_embeddings_aegis_safety_harmful.pt", weights_only=True)
            benign_image_question_embeddings = torch.load("full_dataset_embeddings_aegis_safety_benign.pt", weights_only=True)
        elif args.dataset == "beavertails_rlhf":
            harmful_image_question_embeddings = torch.load("full_dataset_embeddings_beavertails_rlhf_harmful.pt", weights_only=True)
            benign_image_question_embeddings = torch.load("full_dataset_embeddings_beavertails_rlhf_benign.pt", weights_only=True)
        elif args.dataset == "AIAH":
            harmful_image_question_embeddings = torch.load("full_dataset_embeddings_AIAH.pt", weights_only=True)
        else:
            raise NotImplementedError
        
        benign_image_question_embeddings = torch.load('full_dataset_embeddings_alpacaeval_full.pt', weights_only=True)
    
    elif args.use_fixed_main_training_dataset:
        ## the fixed training dataset is the second aegis safety dataset
        benign_image_question_embeddings = torch.load('full_dataset_embeddings_second_time_aegis_safety_dataset_benign.pt', weights_only=True)
        harmful_image_question_embeddings = torch.load('full_dataset_embeddings_second_time_aegis_safety_dataset_harmful.pt', weights_only=True)
        print(f"Shape of the harmful image question embeddings: {harmful_image_question_embeddings.shape} and benign image question embeddings: {benign_image_question_embeddings.shape}")
    
    harmful_image_question_embeddings = harmful_image_question_embeddings[:, best_layer, :].clone().to(device)
    benign_image_question_embeddings = benign_image_question_embeddings[:, best_layer, :].clone().to(device)
    
    print(f"Shape of the harmful image question embeddings: {harmful_image_question_embeddings.shape} and benign image question embeddings: {benign_image_question_embeddings.shape}, type: {harmful_image_question_embeddings.dtype} and {benign_image_question_embeddings.dtype}")      ## (num_datapoints, layers, hidden_dim)
    
    ## make both the datasets of the same size by upsampling the smaller one
    if harmful_image_question_embeddings.shape[0] > benign_image_question_embeddings.shape[0]:
        diff = harmful_image_question_embeddings.shape[0] - benign_image_question_embeddings.shape[0]
        idx = torch.randint(benign_image_question_embeddings.shape[0], (diff,), device=benign_image_question_embeddings.device)
        # benign_image_question_embeddings = torch.cat([benign_image_question_embeddings, benign_image_question_embeddings[torch.randint(0, benign_image_question_embeddings.shape[0], (harmful_image_question_embeddings.shape[0] - benign_image_question_embeddings.shape[0],))]])
        benign_image_question_embeddings = torch.cat([benign_image_question_embeddings, benign_image_question_embeddings[idx]])
    elif benign_image_question_embeddings.shape[0] > harmful_image_question_embeddings.shape[0]:
        diff = benign_image_question_embeddings.shape[0] - harmful_image_question_embeddings.shape[0]
        idx = torch.randint(harmful_image_question_embeddings.shape[0], (diff,), device=harmful_image_question_embeddings.device)
        # harmful_image_question_embeddings = torch.cat([harmful_image_question_embeddings, harmful_image_question_embeddings[torch.randint(0, harmful_image_question_embeddings.shape[0], (benign_image_question_embeddings.shape[0] - harmful_image_question_embeddings.shape[0],))]])
        harmful_image_question_embeddings = torch.cat([harmful_image_question_embeddings, harmful_image_question_embeddings[idx]])

    print(f"Shape of the harmful image question embeddings: {harmful_image_question_embeddings.shape} and benign image question embeddings: {benign_image_question_embeddings.shape}")      ## (num_datapoints, layers, hidden_dim)
    assert harmful_image_question_embeddings.shape[0] == benign_image_question_embeddings.shape[0], f"Number of images: {harmful_image_question_embeddings.shape[0]} is not equal to the number of captions: {benign_image_question_embeddings.shape[0]}"
    
    ## now create labels for the two datasets, harmful will be 1 and benign will be 0. Take 50% for training, 25% validation, and 25% testing. Train a simple classifier on the embeddings
    all_labels = torch.cat([torch.ones(harmful_image_question_embeddings.shape[0]), torch.zeros(benign_image_question_embeddings.shape[0])])
    all_embeddings = torch.cat([harmful_image_question_embeddings, benign_image_question_embeddings]).float()
    print(f"Shape of the embeddings: {all_embeddings.shape} and labels: {all_labels.shape}")
    
    from torch.utils.data import Dataset, DataLoader, random_split
    
    class CustomDataset(Dataset):
        def __init__(self, embeddings, labels):
            self.embeddings = embeddings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.embeddings[idx], self.labels[idx]
    
    ## now split the dataset into train, validation and test -- use torch for this
    # Split dataset: 50% train, 25% validation, 25% test
    all_dataset = CustomDataset(all_embeddings, all_labels)
    train_size = int(0.7 * len(all_dataset))
    val_size = int(0.15 * len(all_dataset))
    test_size = len(all_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(all_dataset, [train_size, val_size, test_size])
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    if args.load_external_test_dataset is not None:
        ## here we will load the external test dataset
        def load_external_test_dataset(external_test_dataset):
            print(f"Loading external test dataset: {external_test_dataset}")
            external_benign_image_question_embeddings, external_harmful_image_question_embeddings = None, None
            if external_test_dataset == "advbench":
                external_harmful_image_question_embeddings = torch.load('full_dataset_embeddings_advbench.pt', weights_only=True)
            elif external_test_dataset == "AIAH":
                external_harmful_image_question_embeddings = torch.load('full_dataset_embeddings_AIAH.pt', weights_only=True)
            elif external_test_dataset == "forbidden_questions_dataset":
                external_harmful_image_question_embeddings = torch.load('full_dataset_embeddings_forbidden_questions_dataset_harmful.pt', weights_only=True)
            elif external_test_dataset == "harmbench_dataset":
                external_harmful_image_question_embeddings = torch.load('full_dataset_embeddings_harmbench_dataset_harmful.pt', weights_only=True)
            elif external_test_dataset == "saladbench_dataset":
                external_harmful_image_question_embeddings = torch.load('full_dataset_embeddings_saladbench_dataset_harmful.pt', weights_only=True)
            elif external_test_dataset == "simple_safety_dataset":
                external_harmful_image_question_embeddings = torch.load('full_dataset_embeddings_simple_safety_dataset_harmful.pt', weights_only=True)
            elif external_test_dataset == "xsafety_dataset":
                external_harmful_image_question_embeddings = torch.load('full_dataset_embeddings_xsafety_dataset_harmful.pt', weights_only=True)
            elif external_test_dataset == "toxicity_jigsaw_dataset":
                external_harmful_image_question_embeddings = torch.load('full_dataset_embeddings_toxicity_jigsaw_dataset_harmful.pt', weights_only=True)
                external_benign_image_question_embeddings = torch.load('full_dataset_embeddings_toxicity_jigsaw_dataset_benign.pt', weights_only=True)
            elif external_test_dataset == "safebench_male":
                external_harmful_image_question_embeddings = torch.load('full_dataset_embeddings_safebench_male_harmful.pt', weights_only=True)
            elif external_test_dataset == "safebench_female":
                external_harmful_image_question_embeddings = torch.load('full_dataset_embeddings_safebench_female_harmful.pt', weights_only=True)
            else:
                raise NotImplementedError(f"External test dataset {external_test_dataset} not supported")

            if external_harmful_image_question_embeddings is not None:
                external_harmful_image_question_embeddings = external_harmful_image_question_embeddings[:, best_layer, :].clone().to(device)
            if external_benign_image_question_embeddings is not None:
                external_benign_image_question_embeddings = external_benign_image_question_embeddings[:, best_layer, :].clone().to(device)
            
            if external_benign_image_question_embeddings is None and external_harmful_image_question_embeddings is not None:          # no benign samples
                external_benign_image_question_embeddings = external_harmful_image_question_embeddings.new_empty((0, external_harmful_image_question_embeddings.size(1)))
            if external_harmful_image_question_embeddings is None and external_benign_image_question_embeddings is not None:          # no harmful samples
                external_harmful_image_question_embeddings = external_benign_image_question_embeddings.new_empty((0, external_benign_image_question_embeddings.size(1)))
            
            print(f"Shape of the external harmful image question embeddings: {external_harmful_image_question_embeddings.shape} and benign image question embeddings: {external_benign_image_question_embeddings.shape}")
            return external_benign_image_question_embeddings, external_harmful_image_question_embeddings
        
        all_external_test_loaders = []
        for external_dataset_name in args.load_external_test_dataset:
            external_benign_image_question_embeddings, external_harmful_image_question_embeddings = load_external_test_dataset(external_dataset_name)
            ## now create labels for these datasets and dataloaders, and these will be called as external_test_loaders
            external_labels = torch.cat([torch.ones(external_harmful_image_question_embeddings.shape[0]), torch.zeros(external_benign_image_question_embeddings.shape[0])])
            external_embeddings = torch.cat([external_harmful_image_question_embeddings, external_benign_image_question_embeddings]).float()
            assert external_embeddings.shape[0] == external_labels.shape[0], f"Number of images: {external_embeddings.shape[0]} is not equal to the number of captions: {external_labels.shape[0]}"
            print(f"Shape of the external embeddings: {external_embeddings.shape} and labels: {external_labels.shape}")
            ## now create the dataloader for this dataset
            external_dataset = CustomDataset(external_embeddings, external_labels)
            external_test_loader = DataLoader(external_dataset, batch_size=args.batch_size, shuffle=False)
            all_external_test_loaders.append((external_dataset_name, external_test_loader))
    
    # Define a simple feedforward neural network
    import torch.nn as nn
    import torch.optim as optim
    
    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Initialize the model, loss function and optimizer
    input_dim = all_embeddings.shape[1]  # Assuming the last dimension is the feature dimension
    model = SimpleNN(input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model.to(device)

    # Training loop
    best_val_accuracy = 0.0
    best_model = None
    metrics_path = f"external_test_dataset_results_{args.learning_rate}_{args.batch_size}.jsonl"
    # start fresh for this run
    open(metrics_path, "w").close()
    for epoch in range(args.num_epochs):
        model.train()
        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.view(batch_embeddings.size(0), -1).to(device)
            batch_labels = batch_labels.float().to(device)

            optimizer.zero_grad()
            outputs_train = model(batch_embeddings)
            loss_train = criterion(outputs_train.squeeze(), batch_labels)
            loss_train.backward()
            optimizer.step()

        # Validation and testing loop
        model.eval()
        with torch.no_grad():
            def combined_validation_loop(this_dataloader):
                this_loader_loss = 0.0
                this_loader_total = 0
                this_loader_correct = 0
                for batch_embeddings, batch_labels in this_dataloader:
                    batch_embeddings = batch_embeddings.view(batch_embeddings.size(0), -1).to(device)
                    batch_labels = batch_labels.float().to(device)

                    outputs_this_loader = model(batch_embeddings)
                    loss_this_loader = criterion(outputs_this_loader.view(-1), batch_labels)
                    this_loader_loss += loss_this_loader.item()

                    # predicted = torch.round(torch.sigmoid(outputs_this_loader.squeeze()))
                    predicted = torch.round(outputs_this_loader.view(-1).sigmoid())
                    this_loader_total += batch_labels.size(0)
                    this_loader_correct += (predicted == batch_labels).sum().item()

                this_loader_loss /= len(this_dataloader)
                this_loader_accuracy = this_loader_correct * 100.0 / this_loader_total
                return this_loader_loss, this_loader_accuracy

            val_loss, val_accuracy = combined_validation_loop(val_loader)
            test_loss, test_accuracy = combined_validation_loop(test_loader)
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {loss_train.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            if args.load_external_test_dataset is not None:
                external_test_dataset_accuracies = {}
                for external_dataset, external_test_loader in all_external_test_loaders:
                    external_test_loss, external_test_accuracy = combined_validation_loop(external_test_loader)
                    print(f"External Dataset: {external_dataset}, Test Loss: {external_test_loss:.4f}, External Test Accuracy: {external_test_accuracy:.4f}")
                    external_test_dataset_accuracies[external_dataset] = external_test_accuracy
                mean_external_test_accuracy = sum(external_test_dataset_accuracies.values()) / len(external_test_dataset_accuracies)
                print(f"\033[93mMean External Test Accuracy: {mean_external_test_accuracy:.4f}\033[0m")
            
            epoch_metrics = {
                    "epoch": epoch,
                    # ── MAIN (“in-domain”) DATASET ────────────────────────────────
                    "main": {
                        "test_accuracy":            test_accuracy,              # lang → acc
                        "val_accuracy":      val_accuracy,        # lang → acc
                        },
                    # ── EXTERNAL DATASETS ───────────────────────────────────────
                    "external": {
                        "external_test_dataset_accuracies": external_test_dataset_accuracies if args.load_external_test_dataset is not None else None,
                        "mean_external_test_accuracy": mean_external_test_accuracy if args.load_external_test_dataset is not None else None,
                    }
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(epoch_metrics) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answer-file", type=str)
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--input_type", type=str, default="raw")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--s2s", action="store_true", default=False)
    parser.add_argument("--is_lora", action="store_true", default=False)
    parser.add_argument("--measure_cosine_similarity_aligned", action="store_true", default=False)
    parser.add_argument("--measure_cosine_similarity_unaligned", action="store_true", default=False)
    parser.add_argument("--plot_cosine_similarity", action="store_true", default=False)
    parser.add_argument("--compute_image_and_question_embeddings", action="store_true", default=False)
    parser.add_argument("--train_classifier_on_embeddings", action="store_true", default=False)
    parser.add_argument("--use_each_dataset_experimental_design", action="store_true", default=False)
    parser.add_argument("--use_fixed_main_training_dataset", action="store_true", default=False)
    parser.add_argument("--load_external_test_dataset", nargs='+', default=None)
    # Please directly answer the questions in the user's speech.
    args = parser.parse_args()
    if args.plot_cosine_similarity:
        plot_cosine_similarity(args)
    elif args.train_classifier_on_embeddings:
        assert sum([args.use_each_dataset_experimental_design, args.use_fixed_main_training_dataset]) == 1, "Please choose either use_each_dataset_experimental_design or use_fixed_main_training_dataset"
        print(f"using {args.learning_rate} as learning rate and {args.batch_size} as batch size")
        train_classifier_on_embeddings(args)
    else:
        eval_model(args)
