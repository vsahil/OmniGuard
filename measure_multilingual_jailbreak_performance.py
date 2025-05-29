
import os
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, pipeline

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
import yaml, requests, json
from tqdm import tqdm


def load_tokenizer_and_model(args, only_tokenizer=False):
    ## add access token to access the model.
    if args.llm_model == "llama3-8b":
        model_id = "meta-llama/Meta-Llama-3-8B"
    elif args.llm_model == "llama3-8b-instruct":
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif args.llm_model == "llama3-70b":
        model_id = "meta-llama/Meta-Llama-3-70B"
    elif args.llm_model == "llama3-70b-instruct":
        model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    elif args.llm_model == "llama3.1-8b":
        model_id = "meta-llama/Meta-Llama-3.1-8B"
    elif args.llm_model == "llama3.1-8b-instruct":
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif args.llm_model == "llama3.1-70b":
        model_id = "meta-llama/Meta-Llama-3.1-70B"
    elif args.llm_model == "llama3.1-70b-instruct":
        model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    elif args.llm_model == "llama3.3-70b-instruct":
        model_id = "meta-llama/Llama-3.3-70B-Instruct"
    elif args.llm_model == "qwen-2.5-72B-instruct":
        model_id = 'Qwen/Qwen2.5-72B-Instruct'
    elif args.llm_model == "qwen2.5-32b-instruct":
        model_id = 'Qwen/Qwen2.5-32B-Instruct'
    elif args.llm_model == "qwen2.5-0.5b-instruct":
        model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
    elif args.llm_model == "mistral-nemo-12b":
        model_id = "mistralai/Mistral-Nemo-Instruct-2407"
    elif args.llm_model == "multilingual-e5-large":
        model_id = 'intfloat/multilingual-e5-large'
    else:
        raise NotImplementedError

    print(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if only_tokenizer:
        return tokenizer
    
    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "llama3-70b" in args.llm_model or "llama3.1-70b" in args.llm_model or "llama3.3-70b" in args.llm_model or "qwen-2.5-72B-instruct" in args.llm_model:
        num_layers = 80
    elif "qwen2.5-32b" in args.llm_model:
        num_layers = 64
    elif 'mistral-nemo-12b' in args.llm_model:
        num_layers = 40
    elif "llama3-8b" in args.llm_model or "llama3.1-8b" in args.llm_model:
        num_layers = 32
    elif 'multilingual-e5-large' in args.llm_model or "qwen2.5-0.5b" in args.llm_model:
        num_layers = 24      ## this is a contrastive model, so num layers don't matter 
    else:
        raise NotImplementedError
    
    if args.llm_model in ["multilingual-e5-large"]:
        ## here we just return the model
        pipe = AutoModel.from_pretrained(model_id).to(device)
    else:
        if args.get_hidden_layer_representations:
            # pipe = pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto", return_full_text=False, output_hidden_states=True, return_dict_in_generate=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,device_map="auto", output_hidden_states=True,return_dict_in_generate=True)
            model.eval()
            return tokenizer, model, device, num_layers + 1
            # pipe = pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.float64}, device_map="auto", return_full_text=False, output_hidden_states=True, return_dict_in_generate=True)       ## this is for debugging. 
        else:
            pipe = pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto", return_full_text=False)
     
    first_param = next(pipe.model.parameters())
    embed_device = first_param.device
    return tokenizer, pipe, embed_device, num_layers + 1

    ## important files
    # /home/t-vermasahil/miniconda3/envs/llm-defense/lib/python3.11/site-packages/transformers/generation/utils.py - line 2933
    # /home/t-vermasahil/miniconda3/envs/llm-defense/lib/python3.11/site-packages/transformers/pipelines/text_generation.py -- 2 functions changed here.


def get_hidden_layer_representations(args):
    if args.run_on_cluster:
        # blob_directory = "/blob_data/multilingual_jailbreaks/"
        if args.use_random_sampled_toxigen_benign or args.use_random_sampled_toxigen_harmful:
            blob_directory = "/blob_data/random_sampled_toxigen_dataset/"
        elif args.use_length_filtered_aegis_benign or args.use_length_filtered_aegis_harmful:
            blob_directory = "/blob_data/aegis_safety_dataset/"
        elif args.use_jailbreakbench_dataset_benign or args.use_jailbreakbench_dataset_harmful:
            blob_directory = "/blob_data/jailbreakbench_dataset/"
        elif args.use_xstest_dataset_benign or args.use_xstest_dataset_harmful:
            blob_directory = "/blob_data/xstest_dataset/"
        else:
            raise NotImplementedError
        os.makedirs(blob_directory, exist_ok=True)
    else:
        blob_directory = "./"
        if args.use_random_sampled_toxigen_benign or args.use_random_sampled_toxigen_harmful:
            blob_directory = "./toxigen_dataset/"
        elif args.use_length_filtered_aegis_benign or args.use_length_filtered_aegis_harmful:
            blob_directory = "./aegis_safety_dataset/"
        elif args.use_jailbreakbench_dataset_benign or args.use_jailbreakbench_dataset_harmful:
            blob_directory = "./jailbreakbench_dataset/"
        elif args.use_xstest_dataset_benign or args.use_xstest_dataset_harmful:
            blob_directory = "./xstest_dataset/"
        elif args.use_beavertails_rlhf_dataset_benign or args.use_beavertails_rlhf_dataset_harmful:
            blob_directory = "./beavertails_rlhf_dataset/"
        elif args.use_flores200_dataset:
            blob_directory = "./flores200_dataset/"
        elif args.use_mm_vet_dataset or args.use_hades_dataset or args.use_mm_safetybench_dataset or args.use_mm_vet_v2_dataset or args.use_vlsbench_dataset or args.use_mml_safebench_figstep_dataset:
            blob_directory = "./mm_vet_dataset/"
        elif args.use_polyguardmix_train_100K_benign or args.use_polyguardmix_train_100K_harmful or args.use_polyguardmix_train_500K_benign or args.use_polyguardmix_train_500K_harmful or args.use_polyguardmix_test_benign_prompt or args.use_polyguardmix_test_harmful_prompt or args.use_polyguardmix_test_benign_response or args.use_polyguardmix_test_harmful_response or args.use_polyguardmix_train_all_benign or args.use_polyguardmix_train_all_harmful:
            blob_directory = "./polyguardmix_dataset/"
        elif args.use_wildguard_ar or args.use_wildguard_zh or args.use_wildguard_cs or args.use_wildguard_nl or args.use_wildguard_en or args.use_wildguard_fr or args.use_wildguard_de or args.use_wildguard_hi or args.use_wildguard_it or args.use_wildguard_ja or args.use_wildguard_ko or args.use_wildguard_po or args.use_wildguard_pt or args.use_wildguard_ru or args.use_wildguard_es or args.use_wildguard_sv or args.use_wildguard_th or args.use_wildguard_caesar or args.use_wildguard_caesar1 or args.use_wildguard_leet or args.use_wildguard_vowel or args.use_wildguard_base64 or args.use_wildguard_hexadecimal or args.use_wildguard_alphanumeric or args.use_wildguard_ascii:
            blob_directory = "./wildguard_datasets/mWildGuardMix-train-tower-nllb-v2.1/"
        elif args.use_wikidata_en or args.use_wikidata_fr:
            blob_directory = "./wikidata_dataset/"
        elif args.use_filtered_multijail_dataset_en or args.use_filtered_multijail_dataset_ar or args.use_filtered_multijail_dataset_bn or args.use_filtered_multijail_dataset_it or args.use_filtered_multijail_dataset_jv or args.use_filtered_multijail_dataset_ko or args.use_filtered_multijail_dataset_sw or args.use_filtered_multijail_dataset_th or args.use_filtered_multijail_dataset_vi or args.use_filtered_multijail_dataset_zh:
            blob_directory = "./multijail_dataset/"
        elif args.use_filtered_csrt_dataset:
            blob_directory = "./CSRT_dataset/"
        elif args.use_xsafety_bn or args.use_xsafety_fr or args.use_xsafety_sp or args.use_xsafety_zh or args.use_xsafety_ar or args.use_xsafety_hi or args.use_xsafety_ja or args.use_xsafety_ru or args.use_xsafety_de or args.use_xsafety_en:
            blob_directory = "./xsafety_dataset/"
        elif args.use_rtplx_en or args.use_rtplx_others:
            blob_directory = "./rtp_lx_dataset/"
        elif args.use_wildguardmix_sampled_benign or args.use_wildguardmix_sampled_harmful:
            blob_directory = "wildguard_datasets/wildguard-train"
        elif args.use_llm_lat_harmful or args.use_llm_lat_benign:
            blob_directory = "./llm_lat_dataset/"
        elif args.use_aegis2_LG_filtered_benign or args.use_aegis2_LG_filtered_harmful:
            blob_directory = "./Aegis-AI-Content-Safety-Dataset-2.0/llamaguard_filtered_sample/"
        elif args.use_aegis2_random_sample_benign or args.use_aegis2_random_sample_harmful:
            blob_directory = "./Aegis-AI-Content-Safety-Dataset-2.0/random_5K_sample/"
        elif args.use_original_llm_lat_harmful or args.use_original_llm_lat_benign:
            blob_directory = "./llm_lat_dataset/original_llm_lat_dataset/"
        elif args.use_wildguard_word_balanced_benign or args.use_wildguard_word_balanced_harmful:
            blob_directory = "./wildguard_datasets/word_balanced_sample_5K/"
        elif args.use_smaller_aegis_resampled_benign or args.use_smaller_aegis_resampled_harmful:
            blob_directory = "./second_time_aegis_safety_dataset/"
        elif args.use_third_time_aegis_resampled_benign or args.use_third_time_aegis_resampled_harmful:
            blob_directory = "./Aegis-AI-Content-Safety-Dataset-2.0/third_time_aegis2_sampling/"
        elif args.use_oai_moderation_dataset_harmful:
            blob_directory = "./oai_moderation_dataset/"
        elif args.use_harmbench:
            blob_directory = "./harmbench_dataset/"
        elif args.use_forbidden_questions:
            blob_directory = "./forbidden_questions_dataset/"
        elif args.use_filtered_toxicchat_benign or args.use_filtered_toxicchat_harmful:
            blob_directory = "./toxicchat_dataset/"
        elif args.use_simple_safety_tests_dataset:
            blob_directory = "./simple_safety_tests_dataset/"
        elif args.use_saladbench_dataset:
            blob_directory = "./saladbench_dataset/"
        elif args.use_aya_redteaming_dataset:
            blob_directory = "./aya_redteaming_dataset/"
        elif args.use_thai_toxicity_tweets_benign or args.use_thai_toxicity_tweets_harmful:
            blob_directory = "./thai_toxicity_tweets/"
        elif args.use_ukr_tweets_benign or args.use_ukr_tweets_harmful:
            blob_directory = "./ukr_toxicity_dataset/"
        elif args.use_advbench_dataset:
            blob_directory = "./advbench_dataset/"
        elif args.use_toxicity_jigsaw_dataset_benign or args.use_toxicity_jigsaw_dataset_harmful:
            blob_directory = "./toxicity_jigsaw_dataset/"
        elif args.use_toxic_text_dataset_benign or args.use_toxic_text_dataset_harmful:
            blob_directory = "./toxic_text_dataset/"
        elif args.use_code_attack_cipher or args.use_code_attack_python_list or args.use_code_attack_python_stack or args.use_code_attack_python_string:
            blob_directory = "./CodeAttack/harmful_prompts/"
        elif args.use_code_attack_python_list_benign or  args.use_code_attack_python_stack_benign or args.use_code_attack_python_string_benign:
            blob_directory = "./CodeAttack/benign_prompts/"
        elif args.use_code_attack_python_list_benign_testset or args.use_code_attack_python_stack_benign_testset or args.use_code_attack_python_string_benign_testset:
            blob_directory = "./CodeAttack/test_benign_prompts/"
        else:
            raise NotImplementedError
    
    # max_length = 512
    ## this is because we are not running this script in the cluster. Ignore this error.
    if args.add_no_translator_command_english:
        blob_directory = f"{blob_directory}model_representations_no_translator_command_english/"
        no_translate_command = "### Instruction: Do not be a translator. Answer the question in the language it was asked."
        max_length = 530
    elif args.add_no_translator_command_respective_language:
        blob_directory = f"{blob_directory}model_representations_no_translator_command_respective_language/"
        max_length = 530
        no_translate_command_dict = {
                'zh': "\n ### 操作说明： 不要做翻译。用中文回答问题。", 
                'it': "\n ### Istruzioni: Non essere un traduttore. Rispondi alla domanda in italiano.", 
                'vi': "\n ### Chỉ dẫn: Đừng là một dịch giả. Trả lời câu hỏi bằng tiếng Việt.", 
                'ar': "\n ### تعليمات: لا تكن مترجما. أجب عن السؤال باللغة العربية.", 
                'ko': "\n ### 지침: 번역가가 되지 마십시오. 한국어로 질문에 답합니다.", 
                'th': "\n ### คำแนะนำ: อย่าเป็นนักแปล ตอบคําถามเป็นภาษาไทย", 
                'sw': "\n ### Maagizo: Usiwe mtafsiri. Jibu swali hili kwa Kiswahili.", 
                'jv': "\n ### Pituduh: Aja dadi penerjemah. Wangsulana pitakon nganggo basa Jawa.",
                'bn': "\n ### নির্দেশ: অনুবাদক হওয়া না। প্রশ্নের উত্তর দিন বাংলায়।"
        }
    elif args.add_internally_translate_but_output_same_language:
        blob_directory = f"{blob_directory}model_representations_internally_translate_but_output_same_language/"
        max_length = 530
        no_translate_command = "### Instruction: Translate the question internally to english, but give the final answer in the language it was asked."
    else:
        if args.use_toxigen_data:
            blob_directory = f"{blob_directory}model_representations_toxigen/"
        # elif args.use_random_sampled_toxigen_benign or args.use_random_sampled_toxigen_harmful:
        #     blob_directory = f"{blob_directory}model_representations_random_sampled_toxigen/"
        elif args.use_flores200_dataset:
            blob_directory = f"{blob_directory}model_representations/"
        elif args.use_mm_vet_dataset:
            blob_directory = f"{blob_directory}model_representations_mm_vet_dataset/"
        elif args.use_mm_vet_v2_dataset:
            blob_directory = f"{blob_directory}model_representations_mm_vet_v2_dataset/"
        elif args.use_hades_dataset:
            blob_directory = f"{blob_directory}model_representations_hades_dataset/"
        elif args.use_mm_safetybench_dataset:
            blob_directory = f"{blob_directory}model_representations_mm_safetybench_dataset/"
        elif args.use_vlsbench_dataset:
            blob_directory = f"{blob_directory}model_representations_vlsbench_dataset/"
        elif args.use_mml_safebench_figstep_dataset:
            blob_directory = f"{blob_directory}model_representations_mml_safebench_figstep_dataset/"
        else:
            blob_directory = f"{blob_directory}model_representations_multilingual_jailbreaks/"
    
    os.makedirs(blob_directory, exist_ok=True)
    
    if args.jailbreak_prompt:
        assert not args.use_benign_data, "Cannot use benign dataset with jailbreak prompts."
        get_jailbreak_prompts(args)
    
    # assert args.specific_language is not None
    
    if args.use_multijail_data:
        df = pd.read_csv("multijail_results/multijail_dataset.csv")
    
    languages_names = {'en': 'English', 'fr': 'French', 'de': 'German', 'es': 'Spanish', 'fa': 'Farsi', 'ar': 'Arabic', 'hr': 'Croatian', 'ja': 'Japanese', 'pl': 'Polish', 'ru': 'Russian', 'sv': 'Swedish', 'th': 'Thai', 'hi': 'Hindi', 'bn': 'Bengali', 'it': 'Italian', 'ko': 'Korean', 'pt': 'Portuguese', 'te': 'Telugu', 'vi': 'Vietnamese', 'zh': 'Chinese', 'he': 'Hebrew', 'sr': 'Serbian', 'da': 'Danish', 'tr': 'Turkish', 'el': 'Greek', 'id': 'Indonesian', 'nl': 'Dutch', 'cs': 'Czech', 'sp': 'Spanish',
    
    'caesar': 'caesar', 'caesar1': 'caesar1', 'caesar2': 'caesar2', 'caesar4': 'caesar4', 'caesar5': 'caesar5', 'caesar6': 'caesar6', 'caesar7': 'caesar7', 'caesar8': 'caesar8', 'caesar9': 'caesar9', 'caesarneg1': 'caesarneg1', 'caesarneg2': 'caesarneg2', 'caesarneg3': 'caesarneg3', 'caesarneg4': 'caesarneg4', 'caesarneg5': 'caesarneg5', 'caesarneg6': 'caesarneg6', 'caesarneg7': 'caesarneg7', 'caesarneg8': 'caesarneg8', 'caesarneg9': 'caesarneg9', 'base64': 'base64', 'hexadecimal': 'hexadecimal', 'alphanumeric': 'alphanumeric', 'atbash': 'atbash', 'reverse': 'reverse', 'ascii': 'ascii', 'vowel': 'vowel', 'leet': 'leet', 'morse': 'morse'}
    cipher_languages = ['caesar1', 'caesar2', 'caesar', 'caesar4', 'caesar5', 'caesar6', 'caesar7', 'caesarneg1', 'caesarneg2', 'caesarneg3', 'caesarneg4', 'caesarneg5', 'caesarneg6', 'caesarneg7', 'ascii', 'hexadecimal', 'base64', 'leet', 'vowel', 'alphanumeric']
    
    tokenizer, model, device, model_num_layers = None, None, None, None
    # for lang_text in ['en', 'fr', 'de', 'es', 'fa', 'ar', 'hr', 'ja', 'pl', 'ru', 'sv', 'th', 'hi', 'it', 'ko', 'pt', 'te', 'vi', 'zh', 'he', 'sr', 'da', 'tr', 'el', 'id', 'bn', 'leet', 'vowel', 'caesar', 'caesar1', 'caesar2', 'caesar4', 'caesar5', 'caesar6', 'caesarneg1', 'caesarneg2', 'caesarneg3', 'caesarneg4', 'caesarneg5', 'caesarneg6', 'base64', 'hexadecimal', 'alphanumeric', 'unicode', 'utf', 'atbash', 'vigenere', 'keyboard', 'reverse', 'ascii']:
    # for lang_text in ['caesar7', 'caesar8', 'caesar9', 'caesarneg7', 'caesarneg8', 'caesarneg9', 'morse']:
                    # [0,   1,    2,     3,    4,    5,    6,    7,    8,    9,    10,   11,   12,   13,   14,   15,   16,   17,   18,   19,   20,   21,   22,   23,   24,   25,   26,   27,   28,   29,   30,   31,   32,   33,   34,   35,   36,   37,   38,   39,   40,   41,   42,   43,   44,   45,   46,   47,   48,   49,   50,   51,   52,   53,   54,   55,   56,   57,   58,   59,   60,   61,   62,   63]

    for lang_text in ['en', 'fr', 'de', 'es', 'fa', 'ar', 'hr', 'ja', 'pl', 'ru', 'sv', 'th', 'hi', 'it', 'ko', 'pt', 'zh', 'sr', 'tr', 'el', 'id', 'hu',  'bs', 'ms', 'jv', 'cy', 'bg', 'mi', 'af', 'zu', 'si', 'bn', 'gu', 'kn', 'mr', 'ta', 'am', 'te', 'lo', 'hy', 'nl', 'cs', 'no', 'he', 'da', 'eu', 'sw', 'uk', 'ro', 'sl', 'fi', 'is', 'vi', *cipher_languages][args.for_hidden_layer_representations_language_index:]:
                    # [0,            1,           2,          3,            4,          5,        6,       7,           8,         9,         10,        11,           12,            13,          14,             15,          16,          17,        18,      19,   
    # for lang_text in ['ascii', 'hexadecimal', 'base64', 'alphanumeric', 'caesar1', 'caesar2', 'caesar', 'caesar4', 'caesar5', 'caesar6', 'caesar7', 'caesarneg1', 'caesarneg2', 'caesarneg3', 'caesarneg4', 'caesarneg5', 'caesarneg6', 'caesarneg7', 'leet', 'vowel',][args.for_hidden_layer_representations_language_index:]:
        print("Processing language: ", lang_text)
        if args.use_multijail_data:
            if args.jailbreak_prompt:
                output_representation_file = f"{blob_directory}multijail_dataset_representations_{args.jailbreak_prompt}_jailbreak_{args.llm_model}_{lang}.pt"
                generated_output_text_file = f"{blob_directory}multijail_dataset_output_text_{args.jailbreak_prompt}_jailbreak_{args.llm_model}_{lang}.txt"
            else:
                output_representation_file = f"{blob_directory}multijail_dataset_representations_non_jailbreak_{args.llm_model}_{lang}.pt"
                generated_output_text_file = f"{blob_directory}multijail_dataset_output_text_non_jailbreak_{args.llm_model}_{lang}.txt"
        elif args.use_benign_data:
            output_representation_file = f"{blob_directory}benign_dataset_representations_{args.llm_model}_{lang}.pt"
            generated_output_text_file = f"{blob_directory}benign_dataset_output_text_{args.llm_model}_{lang}.txt"
            df = pd.read_csv(f"benign_datasets/benign_dataset_{lang}.csv", sep="###\t###", engine='python')
            assert df.shape == (315, 3)
        elif args.use_toxigen_data:
            ## the first column is inputs and the second column is jailbreak_label. The first 400 rows are jailbreak and the next 400 rows are benign.
            if lang == "leet":
                df = pd.read_csv(f"difficult_sampled_toxigen/difficult_sampled_toxigen_{lang}.csv", sep='Ʃ', engine='python', quoting=3)
            else:
                df = pd.read_csv(f"difficult_sampled_toxigen/difficult_sampled_toxigen_{lang}.csv", sep="|", engine='python', quoting=3)
            assert df.shape == (1000, 2), f"Shape of the dataset is not correct: {df.shape}"
            ## assert the first 400 jailbreak_label are 1 and the next 400 are 0.
            assert all(df['jailbreak_label'][:500] == 1)
            assert all(df['jailbreak_label'][500:] == 0)
            ## create a new file with the df prompts column -- each a new line. R
            # df.to_csv(f"cleaned_toxigen_{lang}.csv", sep="|", index=False, quoting=0)
            output_representation_file = f"{blob_directory}difficult_sampled_toxigen_{lang}_representations_{args.llm_model}_{lang}.pt"
            generated_output_text_file = f"{blob_directory}difficult_sampled_toxigen_{lang}_output_text_{args.llm_model}_{lang}.txt"
        elif args.use_random_sampled_toxigen_benign:
            benign_file = f"toxigen_dataset/benign_{lang_text}.txt"
            with open(benign_file, "r") as f:
                all_inputs = f.readlines()
            assert len(all_inputs) == 1000
            output_representation_file = f"{blob_directory}toxigen_dataset_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            # generated_output_text_file = f"{blob_directory}randomly_sampled_toxigen_{lang_text}_output_text_{args.llm_model}_{lang_text}.txt"
        elif args.use_random_sampled_toxigen_harmful:
            toxic_file = f"toxigen_dataset/harmful_{lang_text}.txt"
            with open(toxic_file, "r") as f:
                all_inputs = f.readlines()
            assert len(all_inputs) == 1000
            output_representation_file = f"{blob_directory}toxigen_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_length_filtered_aegis_benign:
            benign_file = f"aegis_safety_dataset/benign_{lang_text}.txt"
            with open(benign_file, "r") as f:
                all_inputs = f.readlines()
            assert len(all_inputs) == 1268, f"Length of the dataset is not correct: {len(all_inputs)}"
            output_representation_file = f"{blob_directory}aegis_safety_dataset_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_length_filtered_aegis_harmful:
            harmful_file = f"aegis_safety_dataset/harmful_{lang_text}.txt"
            with open(harmful_file, "r") as f:
                all_inputs = f.readlines()
            assert len(all_inputs) == 1268, f"Length of the dataset is not correct: {len(all_inputs)}"
            output_representation_file = f"{blob_directory}aegis_safety_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_jailbreakbench_dataset_benign:
            benign_file = f"jailbreakbench_dataset/benign_{lang_text}.txt"
            with open(benign_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) == 100
            output_representation_file = f"{blob_directory}jailbreakbench_dataset_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_jailbreakbench_dataset_harmful:
            harmful_file = f"jailbreakbench_dataset/harmful_{lang_text}.txt"
            with open(harmful_file, "r") as f:
                all_inputs = f.readlines()
            assert len(all_inputs) == 100
            output_representation_file = f"{blob_directory}jailbreakbench_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_xstest_dataset_benign:
            benign_file = f"xstest_dataset/benign_{lang_text}.txt"
            with open(benign_file, "r") as f:
                all_inputs = f.readlines()
            assert len(all_inputs) == 200
            output_representation_file = f"{blob_directory}xstest_dataset_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_xstest_dataset_harmful:
            harmful_file = f"xstest_dataset/harmful_{lang_text}.txt"
            with open(harmful_file, "r") as f:
                all_inputs = f.readlines()
            assert len(all_inputs) == 200
            output_representation_file = f"{blob_directory}xstest_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_beavertails_rlhf_dataset_benign:
            benign_file = f"beavertails_rlhf_dataset/benign_{lang_text}.txt"
            with open(benign_file, "r") as f:
                all_inputs = f.readlines()
            assert len(all_inputs) == 2000
            output_representation_file = f"{blob_directory}beavertails_rlhf_dataset_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_beavertails_rlhf_dataset_harmful:
            harmful_file = f"beavertails_rlhf_dataset/harmful_{lang_text}.txt"
            with open(harmful_file, "r") as f:
                all_inputs = f.readlines()
            assert len(all_inputs) == 2000
            output_representation_file = f"{blob_directory}beavertails_rlhf_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_flores200_dataset:
            benign_file = f"flores200_dataset/benign_{lang_text}.txt"
            with open(benign_file, "r") as f:
                all_inputs = f.readlines()
            assert len(all_inputs) == 2009
            output_representation_file = f"{blob_directory}flores200_dataset_benign_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_mm_vet_dataset:
            benign_file = "/gscratch/zlab/vsahil/safety_multimodal/multimodal_linkage_attack_paper/generated_captions_mm_vet.json"
            with open(benign_file, "r") as f:
                all_inputs = json.load(f)
            assert len(all_inputs) == 218, f"Length of the dataset is not correct: {len(all_inputs)}"
            ## each item in the json file has the fields: "txt", "img", "image_caption". Extract the "txt" and "img_caption" fields, concatenate them and store in the all_inputs list.
            all_inputs = [f"Answer this question given the image description: Question: {item['txt']}. Image description: {item['image_caption']}" for item in all_inputs]
            output_representation_file = f"{blob_directory}mm_vet_dataset_benign_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_mm_vet_v2_dataset:
            benign_file = "/gscratch/zlab/vsahil/safety_multimodal/multimodal_linkage_attack_paper/generated_captions_mm_vet2.json"
            with open(benign_file, "r") as f:
                all_inputs = json.load(f)
            assert len(all_inputs) == 517, f"Length of the dataset is not correct: {len(all_inputs)}"
            ## each item in the json file is a key whose values has the following fields: "txt", "img", "image_caption". Extract the "txt" and "img_caption" fields, concatenate them and store in the all_inputs list.
            all_inputs = [f"Answer this question given the image description: Question: {all_inputs[question_id]['txt']}. Image description: {all_inputs[question_id]['image_caption']}" for question_id in sorted(all_inputs.keys())]
            output_representation_file = f"{blob_directory}mm_vet_v2_dataset_benign_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_vlsbench_dataset:
            data_root = "/gscratch/zlab/vsahil/safety_multimodal/multimodal_linkage_attack_paper/VLSBench/data"
            data = json.load(open(os.path.join(data_root, "data.json"), 'r'))
            assert len(data) == 2241, f"Length of the dataset is not correct: {len(data)}"
            all_inputs = [f"Answer this question given the image description: Question: {item['instruction']}. Image description: {item['image_description']}" for item in data]
            output_representation_file = f"{blob_directory}vlsbench_dataset_benign_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_mml_safebench_figstep_dataset:
            benign_file = f"/gscratch/zlab/vsahil/safety_multimodal/multimodal_linkage_attack_paper/generated_captions_mml_safebench_images_figstep_{args.mml_safebench_category_index}.json"
            with open(benign_file, "r") as f:
                all_inputs = json.load(f)
            all_inputs = [f"Answer this question given the image description: Question: {item['question']}. Image description: {item['image_caption']}" for item in all_inputs]
            output_representation_file = f"{blob_directory}mml_safebench_figstep_dataset_{args.mml_safebench_category_index}_benign_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_hades_dataset:
            generated_captions = json.load(open("/gscratch/zlab/vsahil/safety_multimodal/multimodal_linkage_attack_paper/generated_captions_hades.json", 'r'))
            ## each element in generated_caption has two fields: question and caption, make them in a list new_test_split: so new_test_split be a list of tuples with (question_id, question, caption). Do it in one line. Note that generated_captions does not have "id" field, so we need to add it.
            all_inputs = [(item['question'], item['image_caption']) for _, item in enumerate(generated_captions)]
            all_inputs = [f"Answer this question given the image description: Question: {item[0]}. Image description: {item[1]}" for item in all_inputs]
            output_representation_file = f"{blob_directory}hades_dataset_benign_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_mm_safetybench_dataset:
            filename = f"/gscratch/zlab/vsahil/safety_multimodal/multimodal_linkage_attack_paper/generated_captions_mm_safetybench_{args.mm_safetybench_category_index}.json"      ## each item here has two fields: question and image_caption
            all_inputs = json.load(open(filename, 'r'))     # image_description = questions[question_id]["image_caption"]   # question = questions[question_id]["question"]
            all_inputs = [f"Answer this question given the image description: Question: {all_inputs[question_id]['question']}. Image description: {all_inputs[question_id]['image_caption']}" for question_id in all_inputs]
            output_representation_file = f"{blob_directory}mm_safetybench_dataset_{args.mm_safetybench_category_index}_benign_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_wildguard_ar or args.use_wildguard_zh or args.use_wildguard_cs or args.use_wildguard_nl or args.use_wildguard_en or args.use_wildguard_fr or args.use_wildguard_de or args.use_wildguard_hi or args.use_wildguard_it or args.use_wildguard_ja or args.use_wildguard_ko or args.use_wildguard_po or args.use_wildguard_pt or args.use_wildguard_ru or args.use_wildguard_es or args.use_wildguard_sv or args.use_wildguard_th or args.use_wildguard_caesar or args.use_wildguard_caesar1 or args.use_wildguard_leet or args.use_wildguard_vowel or args.use_wildguard_base64 or args.use_wildguard_hexadecimal or args.use_wildguard_alphanumeric or args.use_wildguard_ascii:
            ## the language text is the last two letters of the file name. So get the last two letters of the file name, don't build a condition for each individually, do it in one line
            supported = ["caesar", "caesar1", 'leet', "vowel", "base64", "hexadecimal", "alphanumeric", "ascii", "ar", "zh", "cs", "nl", "en", "fr", "de", "hi", "it", "ja", "ko", "po", "pt", "ru", "es", "sv", "th"]
            lang_of_parquet = next((lg for lg in supported if getattr(args, f"use_wildguard_{lg}", False)), None)
            parquet_file = f"wildguard_datasets/mWildGuardMix-train-tower-nllb-v2.1/wildguard-train-{languages_names[lang_of_parquet]}.parquet"
            ## load the parquet file and the get prompts as a list of all_inputs
            df = pd.read_parquet(parquet_file)
            ## assert that the id column is sorted in strictly increasing order.
            assert df['id'].is_monotonic_increasing, "The id column is not sorted in strictly increasing order."
            assert df['id'].max() == len(df) - 1, "The id column is not in the range [0, len(df) - 1]."
            assert df['id'].min() == 0, "The id column is not in the range [0, len(df) - 1]."
            all_inputs = df['prompt'].tolist()
            assert len(all_inputs) == 86759
            output_representation_file = f"{blob_directory}wildguard_dataset_{lang_of_parquet}_representations_{args.llm_model}_{lang_of_parquet}.pt"
        elif args.use_wikidata_en or args.use_wikidata_fr:
            input_file = "finding_different_model_parts_experiment/old_wikitext_translated.txt"
            delimiter = "&*&*&"
            with open(input_file, "r", encoding="utf-8") as file:
                rows = [line.strip().split(delimiter) for line in file]
            headers = rows[0]  # First row
            data_rows = rows[1:]  # Remaining rows
            columns = list(zip(*data_rows))
            inputs_languages = {}
            for i, column in enumerate(columns):
                header_name = headers[i].strip().replace(" ", "_")      # Replace spaces with underscores
                header_name = header_name.replace('"', '').replace("'", "")     # Remove quotes
                cleaned_column = [
                    row.strip('"') if row.startswith('"') and row.endswith('"') else row
                    for row in column
                ]
                inputs_languages[header_name] = cleaned_column
            print(inputs_languages.keys())
            supported = list(inputs_languages.keys())
            lang_text = next((lg for lg in supported if getattr(args, f"use_wikidata_{lg}", False)), None)
            all_inputs = inputs_languages[lang_text]
            assert len(all_inputs) == 1000
            output_representation_file = f"{blob_directory}wikidata_dataset_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            print(f"Loaded wikitext dataset for {lang_text} with {len(all_inputs)} samples.")
        elif args.use_polyguardmix_train_100K_benign or args.use_polyguardmix_train_100K_harmful or args.use_polyguardmix_test_benign_prompt or args.use_polyguardmix_test_harmful_prompt or args.use_polyguardmix_test_benign_response or args.use_polyguardmix_test_harmful_response or args.use_polyguardmix_train_500K_benign or args.use_polyguardmix_train_500K_harmful or args.use_polyguardmix_train_all_benign or args.use_polyguardmix_train_all_harmful:
            if args.use_polyguardmix_train_100K_benign or args.use_polyguardmix_train_100K_harmful or args.use_polyguardmix_train_500K_benign or args.use_polyguardmix_train_500K_harmful or args.use_polyguardmix_train_all_benign or args.use_polyguardmix_train_all_harmful:
                files_directory = "PolyGuardMix/data"
            elif args.use_polyguardmix_test_benign_prompt or args.use_polyguardmix_test_harmful_prompt or args.use_polyguardmix_test_benign_response or args.use_polyguardmix_test_harmful_response:
                files_directory = "PolyGuardPrompts/data"
            ## there are .parquet files in the files_directory, read them and print their shapes
            all_dfs = []
            for file in os.listdir(files_directory):
                if file.endswith(".parquet"):
                    df = pd.read_parquet(os.path.join(files_directory, file))
                    print(f"Shape of {file}: {df.shape}")
                    ## if there are any rows where prompt is null or just whitespace, drop them.
                    df = df.dropna(subset=['prompt'])
                    df = df[df['prompt'].str.strip() != '']
                    print(f"Shape of {file} after dropping nulls: {df.shape}")
                    all_dfs.append(df)
            ## concatenate all of them and take 100K random samples from the dataframe.
            df = pd.concat(all_dfs)
            all_inputs = []
            if args.use_polyguardmix_train_100K_benign or args.use_polyguardmix_train_500K_benign or args.use_polyguardmix_train_all_benign:
                ## we will make 70% of the dataset for prompts and 30% for responses. So create two dataframes, one where column 'prompt_harm_label' is no and another where response_harm_label is no.
                df_prompt = df[df['prompt_harm_label'] == 'no']
                df_response = df[(df['response_harm_label'] == 'no')]
                ## drop any rows from df_response if the response is null or just whitespace.
                df_response = df_response.dropna(subset=['response'])
                df_response = df_response[df_response['response'].str.strip() != '']
                # df_weird_response = df[(df['response_harm_label'] == 'no') & (df['response_refusal_label'] == 'no')]
                # print(f"shape of weird response: {df_weird_response.shape}")
                if args.use_polyguardmix_train_100K_benign:
                    ## take 35K rows from df_prompt and 15K rows from df_response. 
                    df_prompt = df_prompt.sample(n=35000, random_state=42)
                    df_response = df_response.sample(n=15000, random_state=42)
                    output_representation_file = f"{blob_directory}polyguardmix_train_100K_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
                elif args.use_polyguardmix_train_500K_benign:
                    ## take 175K rows from df_prompt and 75K rows from df_response. 
                    df_prompt = df_prompt.sample(n=175000, random_state=42)                    
                    df_response = df_response.sample(n=75000, random_state=42)
                    output_representation_file = f"{blob_directory}polyguardmix_train_500K_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
                    
            elif args.use_polyguardmix_train_100K_harmful or args.use_polyguardmix_train_500K_harmful or args.use_polyguardmix_train_all_harmful:
                ## we will make 70% of the dataset for prompts and 30% for responses. So create two dataframes, one where column 'prompt_harm_label' is yes and another where response_harm_label is yes.
                df_prompt = df[df['prompt_harm_label'] == 'yes']
                df_response = df[(df['response_harm_label'] == 'yes')]
                df_response = df_response.dropna(subset=['response'])
                df_response = df_response[df_response['response'].str.strip() != '']
                # df_weird_response = df[(df['response_harm_label'] == 'yes') & (df['response_refusal_label'] == 'yes')]
                # print(f"shape of weird response: {df_weird_response.shape}")
                if args.use_polyguardmix_train_100K_harmful:
                    ## take 35K rows from df_prompt and 15K rows from df_response. 
                    df_prompt = df_prompt.sample(n=35000, random_state=42)
                    df_response = df_response.sample(n=15000, random_state=42)
                    output_representation_file = f"{blob_directory}polyguardmix_train_100K_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
                elif args.use_polyguardmix_train_500K_harmful:
                    ## take 175K rows from df_prompt and 75K rows from df_response. 
                    df_prompt = df_prompt.sample(n=175000, random_state=42)                    
                    df_response = df_response.sample(n=75000, random_state=42)
                    output_representation_file = f"{blob_directory}polyguardmix_train_500K_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"

            elif args.use_polyguardmix_test_benign_prompt:
                df_prompt = df[df['prompt_harm_label'] == 'unharmful']
                ## make an empty df_response with same columns as df_prompt but no rows.
                df_response = pd.DataFrame(columns=df_prompt.columns)
                output_representation_file = f"{blob_directory}polyguardmix_test_benign_prompt_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            
            elif args.use_polyguardmix_test_harmful_prompt:
                df_prompt = df[df['prompt_harm_label'] == 'harmful']
                df_response = pd.DataFrame(columns=df_prompt.columns)
                output_representation_file = f"{blob_directory}polyguardmix_test_harmful_prompt_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            
            elif args.use_polyguardmix_test_benign_response:
                df_response = df[(df['response_harm_label'] == 'unharmful')]
                df_response = df_response.dropna(subset=['response'])
                df_response = df_response[df_response['response'].str.strip() != '']
                df_prompt = pd.DataFrame(columns=df_response.columns)
                output_representation_file = f"{blob_directory}polyguardmix_test_benign_response_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            
            elif args.use_polyguardmix_test_harmful_response:
                df_response = df[(df['response_harm_label'] == 'harmful')]
                df_response = df_response.dropna(subset=['response'])
                df_response = df_response[df_response['response'].str.strip() != '']
                df_prompt = pd.DataFrame(columns=df_response.columns)
                output_representation_file = f"{blob_directory}polyguardmix_test_harmful_response_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            
            else:
                raise NotImplementedError
            
            all_inputs = df_prompt['prompt'].to_list() + df_response['response'].to_list()
            
            if args.use_polyguardmix_train_all_benign or args.use_polyguardmix_train_all_harmful:
                print(f"length of all inputs: {len(all_inputs)} out of which {len(df_prompt['prompt'].to_list())} are prompts and {len(df_response['response'].to_list())} are responses. Total dataseet size is {df.shape}")
                ## we will do a total of 10 chunks for the dataset using the hyperparameter: polyguardmix_train_all_chunk. the first chunk will be 0, the second chunk will be 1 and so on. 
                total_chunks = 10
                size_of_chunk = len(all_inputs) // total_chunks
                start_index = args.polyguardmix_train_all_chunk * size_of_chunk
                if args.polyguardmix_train_all_chunk == total_chunks - 1:
                    end_index = len(all_inputs)
                else:
                    end_index = (args.polyguardmix_train_all_chunk + 1) * size_of_chunk
                all_inputs = all_inputs[start_index:end_index]
                if args.use_polyguardmix_train_all_benign:
                    output_representation_file = f"{blob_directory}polyguardmix_train_all_benign_chunk_{args.polyguardmix_train_all_chunk}_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
                elif args.use_polyguardmix_train_all_harmful:
                    output_representation_file = f"{blob_directory}polyguardmix_train_all_harmful_chunk_{args.polyguardmix_train_all_chunk}_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            print(f"length of all inputs: {len(all_inputs)}. length of df_prompt: {len(df_prompt['prompt'].to_list())} and length of df_response: {len(df_response['response'].to_list())}.  Output file: {output_representation_file}") 
        elif args.use_filtered_multijail_dataset_en or args.use_filtered_multijail_dataset_ar or args.use_filtered_multijail_dataset_bn or args.use_filtered_multijail_dataset_it or args.use_filtered_multijail_dataset_jv or args.use_filtered_multijail_dataset_ko or args.use_filtered_multijail_dataset_sw or args.use_filtered_multijail_dataset_th or args.use_filtered_multijail_dataset_vi or args.use_filtered_multijail_dataset_zh:
            supported = ["en", "ar", "bn", "it", "jv", "ko", "sw", "th", "vi", "zh"]
            lang_text = next((lg for lg in supported if getattr(args, f"use_filtered_multijail_dataset_{lg}", False)), None)
            filename = f"multijail_dataset/final_multijail_harmful_{lang_text}.csv"
            filename = pd.read_csv(filename)
            assert filename.shape == (275, 3), f"Shape of the dataset is not correct: {filename.shape}"
            all_inputs = filename['text'].to_list()
            assert len(all_inputs) == 275
            output_representation_file = f"{blob_directory}multijail_dataset_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_filtered_csrt_dataset:
            filename = "CSRT_dataset/final_csrt_harmful_code_switch.csv"
            filename = pd.read_csv(filename)
            assert filename.shape == (275, 2), f"Shape of the dataset is not correct: {filename.shape}"
            all_inputs = filename['text'].to_list()
            output_representation_file = f"{blob_directory}csrt_dataset_representations_{args.llm_model}_code_switched.pt"
        elif args.use_xsafety_bn or args.use_xsafety_fr or args.use_xsafety_sp or args.use_xsafety_zh or args.use_xsafety_ar or args.use_xsafety_hi or args.use_xsafety_ja or args.use_xsafety_ru or args.use_xsafety_de or args.use_xsafety_en:
            supported = ["bn", "fr", "sp", "zh", "ar", "hi", "ja", "ru", "de", "en"]
            lang_text = next((lg for lg in supported if getattr(args, f"use_xsafety_{lg}", False)), None)
            filename = f"xsafety_dataset/final_xsafety_harmful_{lang_text}.csv"
            filename = pd.read_csv(filename)
            assert filename.shape <= (417, 3), f"Shape of the dataset is not correct: {filename.shape}"
            all_inputs = filename['text'].to_list()
            output_representation_file = f"{blob_directory}xsafety_dataset_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_rtplx_en or args.use_rtplx_others:
            ## filenames are under rtp_lx_dataset/harmful_indexes_lang.csv. for en use lanf as EN and for others, use all the .csv files that are not EN
            if args.use_rtplx_en:
                filename = [f"rtp_lx_dataset/harmful_indexes_EN.csv"]
                lang_text = "en"
            elif args.use_rtplx_others:
                ## find all .csv files in the directory and get their names
                other_csv_files = [f for f in os.listdir("rtp_lx_dataset") if f.endswith(".csv") and f != "harmful_indexes_EN.csv"]
                filename = [f"rtp_lx_dataset/{f}" for f in other_csv_files]
                lang_text = "others"
            all_inputs = []
            for file in filename:
                df = pd.read_csv(file)
                print(f"Shape of {file}: {df.shape}")
                all_inputs += df['Prompt'].to_list()
            print(f"Length of all inputs: {len(all_inputs)}")
            output_representation_file = f"{blob_directory}rtp_lx_dataset_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_wildguardmix_sampled_benign or args.use_wildguardmix_sampled_harmful:
            if args.use_wildguardmix_sampled_benign:
                filename = f"{blob_directory}/benign_non_adversarial.txt"
            elif args.use_wildguardmix_sampled_harmful:
                filename = f"{blob_directory}/harmful_non_adversarial.txt"
            with open(filename, "r") as f:
                all_inputs = f.readlines()
            assert len(all_inputs) == 19371 if args.use_wildguardmix_sampled_benign else 19498
            all_inputs = [prompt.strip() for prompt in all_inputs]
            output_representation_file = f"{blob_directory}wildguardmix_sampled_{'benign' if args.use_wildguardmix_sampled_benign else 'harmful'}_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
        elif args.use_llm_lat_harmful or args.use_llm_lat_benign:
            if args.use_llm_lat_benign:
                relevant_file = f"llm_lat_dataset/all_final_prompts/benign_{lang_text}.txt"
            else:
                relevant_file = f"llm_lat_dataset/all_final_prompts/harmful_{lang_text}.txt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            if args.use_llm_lat_benign:
                assert len(all_inputs) == 10982, f"Length of the dataset is not correct: {len(all_inputs)}"
                output_representation_file = f"{blob_directory}llm_lat_dataset_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            else:
                assert len(all_inputs) == 10999, f"Length of the dataset is not correct: {len(all_inputs)}"
                output_representation_file = f"{blob_directory}llm_lat_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"

        elif args.use_aegis2_LG_filtered_benign or args.use_aegis2_LG_filtered_harmful:
            if args.use_aegis2_LG_filtered_benign:
                relevant_file = f"Aegis-AI-Content-Safety-Dataset-2.0/llamaguard_filtered_sample/benign_{lang_text}.txt"
                output_representation_file = f"{blob_directory}aegis2_dataset_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            elif args.use_aegis2_LG_filtered_harmful:
                relevant_file = f"Aegis-AI-Content-Safety-Dataset-2.0/llamaguard_filtered_sample/harmful_{lang_text}.txt"
                output_representation_file = f"{blob_directory}aegis2_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) in [4995, 4999], f"Length of the dataset is not correct: {len(all_inputs)}"
        
        elif args.use_aegis2_random_sample_benign or args.use_aegis2_random_sample_harmful:
            if args.use_aegis2_random_sample_benign:
                relevant_file = f"Aegis-AI-Content-Safety-Dataset-2.0/random_5K_sample/benign_{lang_text}.txt"
                output_representation_file = f"{blob_directory}aegis2_dataset_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            elif args.use_aegis2_random_sample_harmful:
                relevant_file = f"Aegis-AI-Content-Safety-Dataset-2.0/random_5K_sample/harmful_{lang_text}.txt"
                output_representation_file = f"{blob_directory}aegis2_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            ## remove lines that are empty of just whitespace
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) in [4997, 5000], f"Length of the dataset is not correct: {len(all_inputs)}"
        
        elif args.use_original_llm_lat_harmful or args.use_original_llm_lat_benign:
            if args.use_original_llm_lat_benign:
                relevant_file = f"llm_lat_dataset/original_llm_lat_dataset/benign_{lang_text}.txt"
                output_representation_file = f"{blob_directory}original_llm_lat_dataset_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            elif args.use_original_llm_lat_harmful:
                relevant_file = f"llm_lat_dataset/original_llm_lat_dataset/harmful_{lang_text}.txt"
                output_representation_file = f"{blob_directory}original_llm_lat_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) in [4947], f"Length of the dataset is not correct: {len(all_inputs)}"

        elif args.use_wildguard_word_balanced_benign or args.use_wildguard_word_balanced_harmful:
            if args.use_wildguard_word_balanced_benign:
                relevant_file = f"wildguard_datasets/word_balanced_sample_5K/benign_{lang_text}.txt"
                output_representation_file = f"{blob_directory}wildguard_balanced_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            elif args.use_wildguard_word_balanced_harmful:
                relevant_file = f"wildguard_datasets/word_balanced_sample_5K/harmful_{lang_text}.txt"
                output_representation_file = f"{blob_directory}wildguard_balanced_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) in [5596, 5597], f"Length of the dataset is not correct: {len(all_inputs)}"

        elif args.use_smaller_aegis_resampled_benign or args.use_smaller_aegis_resampled_harmful:
            if args.use_smaller_aegis_resampled_benign:
                relevant_file = f"second_time_aegis_safety_dataset/benign_{lang_text}.txt"
                output_representation_file = f"{blob_directory}smaller_aegis_resampled_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            elif args.use_smaller_aegis_resampled_harmful:
                relevant_file = f"second_time_aegis_safety_dataset/harmful_{lang_text}.txt"
                output_representation_file = f"{blob_directory}smaller_aegis_resampled_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) in [1426, 1432], f"Length of the dataset is not correct: {len(all_inputs)}"
            
        elif args.use_third_time_aegis_resampled_benign or args.use_third_time_aegis_resampled_harmful:
            if args.use_third_time_aegis_resampled_benign:
                relevant_file = f"Aegis-AI-Content-Safety-Dataset-2.0/third_time_aegis2_sampling/benign_{lang_text}.txt"
                output_representation_file = f"{blob_directory}third_time_aegis_resampled_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            elif args.use_third_time_aegis_resampled_harmful:
                relevant_file = f"Aegis-AI-Content-Safety-Dataset-2.0/third_time_aegis2_sampling/harmful_{lang_text}.txt"
                output_representation_file = f"{blob_directory}third_time_aegis_resampled_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) in [4971, 4999], f"Length of the dataset is not correct: {len(all_inputs)}"

        elif args.use_oai_moderation_dataset_harmful:
            relevant_file = f"oai_moderation_dataset/harmful_{lang_text}.txt"
            output_representation_file = f"{blob_directory}oai_moderation_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) == 522, f"Length of the dataset is not correct: {len(all_inputs)}"
        
        elif args.use_harmbench:
            relevant_file = f"harmbench_dataset/harmful_{lang_text}.txt"
            output_representation_file = f"{blob_directory}harmbench_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) == 224, f"Length of the dataset is not correct: {len(all_inputs)}"
            
        elif args.use_advbench_dataset:
            relevant_file = f"advbench_dataset/harmful_{lang_text}.txt"
            output_representation_file = f"{blob_directory}advbench_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) == 520, f"Length of the dataset is not correct: {len(all_inputs)}"

        elif args.use_forbidden_questions:
            relevant_file = f"forbidden_questions_dataset/harmful_{lang_text}.txt"
            output_representation_file = f"{blob_directory}forbidden_questions_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) == 240, f"Length of the dataset is not correct: {len(all_inputs)}"

        elif args.use_filtered_toxicchat_benign or args.use_filtered_toxicchat_harmful:
            if args.use_filtered_toxicchat_benign:
                relevant_file = f"toxicchat_dataset/benign_{lang_text}.txt"
                output_representation_file = f"{blob_directory}toxicchat_dataset_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            elif args.use_filtered_toxicchat_harmful:
                relevant_file = f"toxicchat_dataset/harmful_{lang_text}.txt"
                output_representation_file = f"{blob_directory}toxicchat_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) in [250, 247], f"Length of the dataset is not correct: {len(all_inputs)}"
            
        elif args.use_simple_safety_tests_dataset:
            relevant_file = f"simple_safety_tests_dataset/harmful_{lang_text}.txt"
            output_representation_file = f"{blob_directory}simple_safety_tests_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) == 100, f"Length of the dataset is not correct: {len(all_inputs)}"
            
        elif args.use_saladbench_dataset:
            relevant_file = f"saladbench_dataset/harmful_{lang_text}.txt"
            output_representation_file = f"{blob_directory}saladbench_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) == 1001, f"Length of the dataset is not correct: {len(all_inputs)}"
            
        elif args.use_aya_redteaming_dataset:
            relevant_file = f"aya_redteaming_dataset/harmful_aya_{lang_text}.txt"
            output_representation_file = f"{blob_directory}aya_redteaming_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            file_lengths = {"en": 302, "fr": 246, "ru": 440, "es": 249, "ar": 355, "hi": 292, "sr": 493, "tl": 285}
            assert len(all_inputs) == file_lengths[lang_text], f"Length of the dataset is not correct: {len(all_inputs)}"
            
        elif args.use_thai_toxicity_tweets_benign or args.use_thai_toxicity_tweets_harmful:
            if args.use_thai_toxicity_tweets_benign:
                relevant_file = f"thai_toxicity_tweets/benign_{lang_text}.txt"
                output_representation_file = f"{blob_directory}thai_toxicity_tweets_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            elif args.use_thai_toxicity_tweets_harmful:
                relevant_file = f"thai_toxicity_tweets/harmful_{lang_text}.txt"
                output_representation_file = f"{blob_directory}thai_toxicity_tweets_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) in [1101, 705], f"Length of the dataset is not correct: {len(all_inputs)}"

        elif args.use_ukr_tweets_benign or args.use_ukr_tweets_harmful:
            if args.use_ukr_tweets_benign:
                relevant_file = f"ukr_toxicity_dataset/benign_uk.txt"
                output_representation_file = f"{blob_directory}ukr_toxicity_dataset_benign_uk_representations_{args.llm_model}_uk.pt"
            elif args.use_ukr_tweets_harmful:
                relevant_file = f"ukr_toxicity_dataset/harmful_uk.txt"
                output_representation_file = f"{blob_directory}ukr_toxicity_dataset_harmful_uk_representations_{args.llm_model}_uk.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) in [1000, 1006], f"Length of the dataset is not correct: {len(all_inputs)}"

        elif args.use_toxicity_jigsaw_dataset_benign or args.use_toxicity_jigsaw_dataset_harmful:
            if args.use_toxicity_jigsaw_dataset_benign:
                relevant_file = f"toxicity_jigsaw_dataset/benign_{lang_text}.txt"
                output_representation_file = f"{blob_directory}toxicity_jigsaw_dataset_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            elif args.use_toxicity_jigsaw_dataset_harmful:
                relevant_file = f"toxicity_jigsaw_dataset/harmful_{lang_text}.txt"
                output_representation_file = f"{blob_directory}toxicity_jigsaw_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) in [658, 689], f"Length of the dataset is not correct: {len(all_inputs)}"

        elif args.use_toxic_text_dataset_benign or args.use_toxic_text_dataset_harmful:
            if args.use_toxic_text_dataset_benign:
                relevant_file = f"toxic_text_dataset/benign_{lang_text}.txt"
                output_representation_file = f"{blob_directory}toxic_text_dataset_benign_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            elif args.use_toxic_text_dataset_harmful:
                relevant_file = f"toxic_text_dataset/harmful_{lang_text}.txt"
                output_representation_file = f"{blob_directory}toxic_text_dataset_harmful_{lang_text}_representations_{args.llm_model}_{lang_text}.pt"
            with open(relevant_file, "r") as f:
                all_inputs = f.readlines()
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
            assert len(all_inputs) in [997, 998], f"Length of the dataset is not correct: {len(all_inputs)}"

        elif args.use_code_attack_cipher or args.use_code_attack_python_list or args.use_code_attack_python_stack or args.use_code_attack_python_string or args.use_code_attack_python_list_benign or  args.use_code_attack_python_stack_benign or args.use_code_attack_python_string_benign or args.use_code_attack_python_list_benign_testset or args.use_code_attack_python_stack_benign_testset or args.use_code_attack_python_string_benign_testset:
            if args.use_code_attack_cipher:
                harmful_file = "./CodeAttack/harmful_prompts/data_ciphers.json"
                output_representation_file = f"{blob_directory}code_attack_cipher_code_representations_{args.llm_model}_code.pt"
            elif args.use_code_attack_python_list:
                harmful_file = "./CodeAttack/harmful_prompts/data_python_list_full.json"
                output_representation_file = f"{blob_directory}code_attack_python_list_code_representations_{args.llm_model}_code.pt"
            elif args.use_code_attack_python_stack:
                harmful_file = "./CodeAttack/harmful_prompts/data_python_stack_full.json"
                output_representation_file = f"{blob_directory}code_attack_python_stack_code_representations_{args.llm_model}_code.pt"
            elif args.use_code_attack_python_string:
                harmful_file = "./CodeAttack/harmful_prompts/data_python_string_full.json"
                output_representation_file = f"{blob_directory}code_attack_python_string_code_representations_{args.llm_model}_code.pt"
            elif args.use_code_attack_python_list_benign:
                harmful_file = "./CodeAttack/benign_prompts/data_benign_prompts_python_list.json"
                output_representation_file = f"{blob_directory}code_attack_python_list_benign_code_representations_{args.llm_model}_code.pt"
            elif args.use_code_attack_python_stack_benign:
                harmful_file = "./CodeAttack/benign_prompts/data_benign_prompts_python_stack.json"
                output_representation_file = f"{blob_directory}code_attack_python_stack_benign_code_representations_{args.llm_model}_code.pt"
            elif args.use_code_attack_python_string_benign:
                harmful_file = "./CodeAttack/benign_prompts/data_benign_prompts_python_string.json"
                output_representation_file = f"{blob_directory}code_attack_python_string_benign_code_representations_{args.llm_model}_code.pt"
            elif args.use_code_attack_python_list_benign_testset:
                harmful_file = "./CodeAttack/test_benign_prompts/data_benign_en_python_list.json"
                output_representation_file = f"{blob_directory}code_attack_python_list_benign_testset_code_representations_{args.llm_model}_code.pt"
            elif args.use_code_attack_python_stack_benign_testset:
                harmful_file = "./CodeAttack/test_benign_prompts/data_benign_en_python_stack.json"
                output_representation_file = f"{blob_directory}code_attack_python_stack_benign_testset_code_representations_{args.llm_model}_code.pt"
            elif args.use_code_attack_python_string_benign_testset:
                harmful_file = "./CodeAttack/test_benign_prompts/data_benign_en_python_string.json"
                output_representation_file = f"{blob_directory}code_attack_python_string_benign_testset_code_representations_{args.llm_model}_code.pt"
            import json
            with open(harmful_file, 'r') as f:
                data = json.load(f)
            # each line has two keys: "plain_attack" and another "code_wrapped_plain_attack" (for the 3 python files) and "cipher" for the cipher file. Get the code_wrapped_plain_attack/cipher part and return it as the harmful text.
            if args.use_code_attack_cipher:
                all_inputs = [x['cipher'] for x in data]
            else:
                all_inputs = [x['code_wrapped_plain_attack'] for x in data]
            if "benign_testset" in output_representation_file:
                assert len(all_inputs) == 520, f"Length of the benign testset dataset is not correct: {len(all_inputs)}"
            elif "benign" in output_representation_file:
                assert len(all_inputs) == 10, f"Length of the benign dataset is not correct: {len(all_inputs)}"     ## this is for the training part. 
            else:
                assert len(all_inputs) == 520, f"Length of the harmful dataset is not correct: {len(all_inputs)}"
            all_inputs = [prompt.strip() for prompt in all_inputs if prompt.strip()]
    
        else:
            raise ValueError("Please provide the correct dataset to use.")

        if args.last_token_representations:
            ## replace representations by last_token_representations
            output_representation_file = output_representation_file.replace("_representations_", "_last_token_representations_")
            os.makedirs(os.path.dirname(output_representation_file), exist_ok=True) ## make the head directory
        
        if args.offload_specific_layer_representations is not None:
            assert os.path.exists(output_representation_file), f"File {output_representation_file} does not exist. Please run the script without --offload_specific_layer_representations first."
            directory_representations = os.path.dirname(output_representation_file)
            new_directory_layer_representations = os.path.join(directory_representations, "layer_wise_representations")
            os.makedirs(new_directory_layer_representations, exist_ok=True)
            new_output_representation_file = os.path.join(new_directory_layer_representations, os.path.basename(output_representation_file).replace("representations", f"layer_{args.offload_specific_layer_representations}_representations"))
            print(f"new output representation file: {new_output_representation_file}")
            ## now load the model representaion file and store the specific layer representation in a new file.
            saved_model_representations = torch.load(output_representation_file, weights_only=True)
            print(f"loaded model representations from {output_representation_file}, shape: {saved_model_representations.shape}")        ## (len(all_inputs), model_num_layers, model_representations.shape[2])
            assert 0 <= args.offload_specific_layer_representations < saved_model_representations.shape[1], f"offload_specific_layer_representations should be between 0 and {saved_model_representations.shape[1] - 1}, but got {args.offload_specific_layer_representations}"
            specific_layer_representations = saved_model_representations[:, args.offload_specific_layer_representations, :].clone()
            ## create a new directory inside the inner most directory of the save_model_representations -- call it, layer_wise_representations. then replace the name of the last occurence of representations with layer_{args.offload_specific_layer_representations}_representations
            torch.save(specific_layer_representations, new_output_representation_file)
            print(f"saved specific layer representations to {new_output_representation_file}")
            continue
        
        if os.path.exists(output_representation_file):
            print(f"File {output_representation_file} already exists. Skipping.")
            continue
        
        if args.llm_model in []:
            raise NotImplementedError ### in the old implementation we used this, no more
            model_representations = []
            ## we will add the representations for each layer to this list. The final shape of this list will be (num_layers + 1, num_prompts, 1, 4096)     ## we will only take the representation of the last token in the sequence.
            batch_size = 32

            # import ipdb; ipdb.set_trace()
            if args.use_multijail_data:
                if args.llm_model == "llama3-8b":
                    all_inputs = [args.this_jailbreak + prompt if args.jailbreak_prompt else prompt for prompt in df[lang].to_list()]
                    if args.add_no_translator_command_english and lang != 'en':
                        all_inputs = [prompt + no_translate_command for prompt in all_inputs]
                    elif args.add_no_translator_command_respective_language and lang != 'en':
                        all_inputs = [prompt + no_translate_command_dict[lang] for prompt in all_inputs]
                    elif args.add_internally_translate_but_output_same_language:
                        all_inputs = [prompt + no_translate_command for prompt in all_inputs]
                
                elif args.llm_model == "llama3-8b-instruct":
                    ## convert the prompt to a dialog and then pass it to the model. in the format: [{"role": "user", "content": prompt}]
                    if args.add_no_translator_command_english or args.add_no_translator_command_respective_language or args.add_internally_translate_but_output_same_language:
                        raise NotImplementedError
                    if args.jailbreak_prompt:
                        dialogs = [ [{"role": "system", "content": args.this_jailbreak}, {"role": "user", "content": prompt}] for prompt in df[lang]]
                    else:
                        dialogs = [ [{"role": "user", "content": prompt}] for prompt in df[lang]]
                    all_inputs = dialogs
                else:
                    raise NotImplementedError
            
            elif args.use_benign_data:
                if args.llm_model == "llama3-8b":
                    all_inputs = df['inputs'].to_list()
                    if args.add_no_translator_command_english and lang != 'en':
                        all_inputs = [prompt + no_translate_command for prompt in all_inputs]
                    elif args.add_no_translator_command_respective_language and lang != 'en':
                        all_inputs = [prompt + no_translate_command_dict[lang] for prompt in all_inputs]
                elif args.llm_model == "llama3-8b-instruct":
                    all_inputs = [ [{"role": "user", "content": prompt}] for prompt in df['inputs'].to_list()]
                    if args.add_no_translator_command_english or args.add_no_translator_command_respective_language:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            
            else:
                raise ValueError("Please provide the correct dataset to use.")
            
            for i in range(0, len(all_inputs), batch_size):
                with torch.no_grad():  # Disable gradient computation
                    inputs = tokenizer(all_inputs[i:i + batch_size], return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)

                    hidden_states = [outputs.hidden_states[i] for i in range(num_layers + 1)]
                    hidden_states = torch.stack(hidden_states).cpu()

                    # Get the input_ids and attention_mask from the inputs
                    input_ids = inputs['input_ids'].cpu()
                    attention_mask = inputs['attention_mask'].cpu()

                    batch_last_hidden_states = []

                    for seq_idx in range(input_ids.size(0)):
                        # Get the last non-PAD token index
                        seq_len = attention_mask[seq_idx].sum().item()  # Number of non-PAD tokens
                        last_non_pad_index = seq_len - 1  # Index of the last non-PAD token
                        # Get the hidden state of the last non-PAD token
                        last_hidden_state = hidden_states[:, seq_idx, last_non_pad_index, :]
                        batch_last_hidden_states.append(last_hidden_state)
                    
                    # Stack the hidden states of the last non-PAD tokens in the batch
                    batch_last_hidden_states = torch.stack(batch_last_hidden_states).cpu()
                    model_representations.append(batch_last_hidden_states)
                    
                    print(f"Done with batch: {i} of {len(all_inputs)}")
                        
            ## now we have the representation for all prompts. Make sure that the final shape is (num_layers + 1, num_prompts, 4096). Do all computation in CPU to avoid memory issues.
            # model_representations is a list of tensors, each tensor has shape (batch_size, 4096). 
            model_representations = torch.cat(model_representations, dim=0).cpu().transpose(0, 1)
            assert model_representations.shape == (num_layers + 1, df.shape[0], 4096)
            print(model_representations.shape)
            torch.save(model_representations, output_representation_file)

        elif args.llm_model in ["llama3-70b", "llama3-70b-instruct", "llama3.1-70b", "llama3.1-70b-instruct", 'llama3-8b', 'llama3-8b-instruct', 'llama3.1-8b', 'llama3.1-8b-instruct', "llama3.3-70b-instruct", 'mistral-nemo-12b', "multilingual-e5-large", "qwen-2.5-72B-instruct", "qwen2.5-32b-instruct", "qwen2.5-0.5b-instruct"]:
            extra_prompt = ""
            if args.add_no_translator_command_english and lang_text != 'en':
                extra_prompt = no_translate_command
            elif args.add_no_translator_command_respective_language and lang_text != 'en':
                extra_prompt = no_translate_command_dict[lang]
            
            if args.use_multijail_data:
                if args.jailbreak_prompt:
                    assert "{{ prompt }}" in args.this_jailbreak, "Please provide the prompt in the jailbreak prompt."
                    all_inputs = [args.this_jailbreak.replace("{{ prompt }}", prompt + extra_prompt) for prompt in df[lang].to_list()]
                else:
                    all_inputs = [prompt + extra_prompt for prompt in df[lang].to_list()]
            elif args.use_benign_data:
                all_inputs = [prompt + extra_prompt for prompt in df['inputs'].to_list()]
            elif args.use_toxigen_data:
                all_inputs = [prompt for prompt in df['inputs'].to_list()]
            elif args.use_random_sampled_toxigen_benign or args.use_random_sampled_toxigen_harmful:
                assert len(all_inputs) == 1000
            elif args.use_length_filtered_aegis_benign or args.use_length_filtered_aegis_harmful:
                assert len(all_inputs) == 1268, f"Length of the dataset is not correct: {len(all_inputs)}"
            elif args.use_jailbreakbench_dataset_benign or args.use_jailbreakbench_dataset_harmful:
                pass
            elif args.use_xstest_dataset_benign or args.use_xstest_dataset_harmful:
                assert len(all_inputs) == 200
            elif args.use_beavertails_rlhf_dataset_benign or args.use_beavertails_rlhf_dataset_harmful:
                assert len(all_inputs) == 2000
            elif args.use_flores200_dataset:
                assert len(all_inputs) == 2009
            elif args.use_mm_vet_dataset:
                assert len(all_inputs) == 218
            elif args.use_hades_dataset:
                assert len(all_inputs) == 750
            elif args.use_mm_safetybench_dataset or args.use_mm_vet_v2_dataset or args.use_vlsbench_dataset or args.use_mml_safebench_figstep_dataset or args.use_polyguardmix_train_100K_benign or args.use_polyguardmix_train_100K_harmful or args.use_polyguardmix_test_benign_prompt or args.use_polyguardmix_test_harmful_prompt or args.use_polyguardmix_test_benign_response or args.use_polyguardmix_test_harmful_response or args.use_polyguardmix_train_500K_benign or args.use_polyguardmix_train_500K_harmful or args.use_polyguardmix_train_all_benign or args.use_polyguardmix_train_all_harmful or args.use_wildguard_ar or args.use_wildguard_zh or args.use_wildguard_cs or args.use_wildguard_nl or args.use_wildguard_en or args.use_wildguard_fr or args.use_wildguard_de or args.use_wildguard_hi or args.use_wildguard_it or args.use_wildguard_ja or args.use_wildguard_ko or args.use_wildguard_po or args.use_wildguard_pt or args.use_wildguard_ru or args.use_wildguard_es or args.use_wildguard_sv or args.use_wildguard_th or args.use_wildguard_caesar or args.use_wildguard_caesar1 or args.use_wildguard_leet or args.use_wildguard_vowel or args.use_wildguard_base64 or args.use_wildguard_hexadecimal or args.use_wildguard_alphanumeric or args.use_wildguard_ascii or args.use_wikidata_en or args.use_wikidata_fr or args.use_filtered_multijail_dataset_en or args.use_filtered_multijail_dataset_ar or args.use_filtered_multijail_dataset_bn or args.use_filtered_multijail_dataset_it or args.use_filtered_multijail_dataset_jv or args.use_filtered_multijail_dataset_ko or args.use_filtered_multijail_dataset_sw or args.use_filtered_multijail_dataset_th or args.use_filtered_multijail_dataset_vi or args.use_filtered_multijail_dataset_zh or args.use_filtered_csrt_dataset or args.use_xsafety_bn or args.use_xsafety_fr or args.use_xsafety_sp or args.use_xsafety_zh or args.use_xsafety_ar or args.use_xsafety_hi or args.use_xsafety_ja or args.use_xsafety_ru or args.use_xsafety_de or args.use_xsafety_en or args.use_rtplx_en or args.use_rtplx_others or args.use_wildguardmix_sampled_benign or args.use_wildguardmix_sampled_harmful or args.use_llm_lat_harmful or args.use_llm_lat_benign or args.use_aegis2_LG_filtered_benign or args.use_aegis2_LG_filtered_harmful or args.use_aegis2_random_sample_benign or args.use_aegis2_random_sample_harmful or args.use_original_llm_lat_harmful or args.use_original_llm_lat_benign or args.use_wildguard_word_balanced_benign or args.use_wildguard_word_balanced_harmful or args.use_smaller_aegis_resampled_benign or args.use_smaller_aegis_resampled_harmful or args.use_third_time_aegis_resampled_benign or args.use_third_time_aegis_resampled_harmful or args.use_oai_moderation_dataset_harmful or args.use_harmbench or args.use_forbidden_questions or args.use_filtered_toxicchat_benign or args.use_filtered_toxicchat_harmful or args.use_simple_safety_tests_dataset or args.use_saladbench_dataset or args.use_aya_redteaming_dataset or args.use_thai_toxicity_tweets_benign or args.use_thai_toxicity_tweets_harmful or args.use_ukr_tweets_benign or args.use_ukr_tweets_harmful or args.use_advbench_dataset or args.use_toxicity_jigsaw_dataset_benign or args.use_toxicity_jigsaw_dataset_harmful or args.use_toxic_text_dataset_benign or args.use_toxic_text_dataset_harmful or args.use_code_attack_cipher or args.use_code_attack_python_list or args.use_code_attack_python_stack or args.use_code_attack_python_string or args.use_code_attack_python_list_benign or  args.use_code_attack_python_stack_benign or args.use_code_attack_python_string_benign or args.use_code_attack_python_list_benign_testset or args.use_code_attack_python_stack_benign_testset or args.use_code_attack_python_string_benign_testset:
                pass
            else:
                raise ValueError("Please provide the correct dataset to use.")
            
            ## now these are the inputs to the model. We want to get the representations of the tokens (average of all the tokens) in each layer of the model -- if using LLM. If using a contrastive model, then directly get the representations of the last layer.
            generated_model_text = []
            model_representations = []
            
            from torch.utils.data import Dataset, DataLoader
            # Custom Dataset Class
            class MyDataset(Dataset):
                def __init__(self, all_inputs):
                    self.all_inputs = all_inputs

                def __len__(self):
                    return len(self.all_inputs)

                def __getitem__(self, i):
                    return self.all_inputs[i]
            
            if tokenizer is None and model is None:
                tokenizer, model, device, model_num_layers = load_tokenizer_and_model(args)

            if args.llm_model in ["multilingual-e5-large"]:
                tokenized_lengths = [(i, len(tokenizer(all_inputs[i])['input_ids'])) for i in range(len(all_inputs))]
                max_tokenized_length = max([i[1] for i in tokenized_lengths])
            else:
                # Tokenize the inputs and sort by length
                # tokenized_lengths = [(i, len(model.tokenizer(all_inputs[i])['input_ids'])) for i in range(len(all_inputs))]
                # model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
                # model.tokenizer.padding_side = 'left'
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = "left"
            
            if args.llm_model in ["multilingual-e5-large"]:
                sorted_tokenized_lengths = sorted(tokenized_lengths, key=lambda x: x[1], reverse=False)
            else:
                pass
                # sorted_tokenized_lengths = sorted(tokenized_lengths, key=lambda x: x[1])
                # # Sort all_inputs based on the sorted tokenized lengths
                # # all_inputs_sorted = [all_inputs[i[0]] for i in sorted_tokenized_lengths]
                # all_inputs_sorted = all_inputs      ## do not sort the inputs, just use the same order as the inputs.
                # assert len(all_inputs_sorted) == len(all_inputs) == len(sorted_tokenized_lengths)
            
            ## This code now implements three optimization to get faster computations:
            # 1. We are using a batch_size, much larger than 1. 
            # 2. If we encounter a OOM error, we do not start from the beginning, we start from the last index we processed, with a lower batch size.
            # 3. We are sorting the dataset in increasing order of length, so that the OOM error occurs at the end of the dataset, and we are guarateed to not benefit by increasing the batch size again. 
            batch_sizes = [128, 64, 32, 16, 8, 4, 2, 1]
            ## print number of gpus
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            assert torch.cuda.device_count() > 1, "Currently script only supports multi-GPU inference"
            # batch_sizes = [16, 8, 4, 2, 1]
            import time, gc
            start_index = 0
            start_time = time.time()
            def collate_fn(batch):
                return tokenizer(batch, return_tensors="pt", padding=True, padding_side="left", truncation=True, max_length=512)
            
            if args.llm_model in ["multilingual-e5-large"]:
                def average_pool_multilingual_e5(last_hidden_states, attention_mask):
                    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
                    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                
                model_representations = []
                batch_dict = tokenizer(all_inputs, max_length=max_tokenized_length + 1, padding=True, truncation=True, return_tensors='pt')

                for batch_size in batch_sizes:
                    if start_index >= len(all_inputs):
                        break
                    try:
                        print(f"Streaming batch_size={batch_size}")
                        with torch.no_grad():
                            for idx in range(start_index, len(all_inputs), batch_size):
                                this_batch_dict = {k: v[idx:idx + batch_size].to(device) for k, v in batch_dict.items()}
                                outputs = model(**this_batch_dict)
                                embeddings = average_pool_multilingual_e5(outputs.last_hidden_state, this_batch_dict['attention_mask'])
                                print(f"finished datapoint {idx + batch_size} of {len(all_inputs)}")
                                model_representations.append(embeddings.cpu())
                                start_index += embeddings.shape[0]
                                del outputs, embeddings, this_batch_dict
                                torch.cuda.empty_cache()
                                gc.collect()
                    
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"OOM error with batch_size={batch_size}, reducing batch size by half")
                            del this_batch_dict
                            torch.cuda.empty_cache()
                            gc.collect()
                        else:
                            raise e  # Re-raise other runtime errors

            else:
                # for batch_size in batch_sizes:
                #     if start_index >= len(all_inputs_sorted):
                #         break
                #     dataset = MyDataset(all_inputs_sorted[start_index:])
                #     printing_interval = min(50, batch_size)
                #     try:
                #         print("-" * 30)
                #         # import ipdb; ipdb.set_trace()
                #         print(f"Streaming batch_size={batch_size}")
                #         ## print the installed version of transformers and torch and numpy
                #         for out in model(dataset, batch_size=batch_size, max_new_tokens=1, pad_token_id=model.tokenizer.eos_token_id, do_sample=False, temperature=None, top_p=None):
                #             try:
                #                 generated_model_text.append(out[0]['generated_text'])
                #             except:
                #                 print("Error in getting the generated text.")
                            
                #             # import ipdb; ipdb.set_trace()
                #             assert len(out[0]['hidden_states']) == model_num_layers
                #             assert all([hidden_state.shape[0] == 1 and hidden_state.shape[1] == out[0]["input_ids"].shape[1] for hidden_state in out[0]['hidden_states']])

                #             hidden_states = torch.cat(out[0]['hidden_states'], dim=0)
                #             if args.use_flores200_dataset:
                #                 ## separately get the last token representation.
                #                 last_token_activations = hidden_states[:, -1, :]
                #                 assert last_token_activations.shape == (model_num_layers, hidden_states.shape[2])
                #             else:
                #                 # Mask to identify non-pad tokens
                #                 non_pad_mask = out[0]["input_ids"].ne(model.tokenizer.pad_token_id)  # Change eos_token_id to pad_token_id if it differs
                #                 expanded_mask = non_pad_mask.unsqueeze(-1).expand_as(hidden_states)
                #                 ## element wise multiplication of the hidden states with the mask, then sum over the tokens, and divide by the number of tokens.
                #                 masked_hidden_states = hidden_states * expanded_mask
                #                 avg_activations = masked_hidden_states.sum(dim=1) / non_pad_mask.sum(dim=1, keepdim=True).float()
                #                 assert avg_activations.shape == (model_num_layers, avg_activations.shape[1])
                            
                #             ## ORIGINAL IMPLEMENTATION WITH NO CARE FOR PAD TOKENS - WRONG
                #             ## get the activation for each layer averaged over all tokens -- for the toxic we want to get it for all the tokens. 
                #             # avg_activations = [hidden_state.mean(dim=1) for hidden_state in out[0]['hidden_states']]
                #             ## make into a tensor of shape (num_layers, dim)
                #             # avg_activations = torch.stack(avg_activations).squeeze(1)
                            
                #             if args.use_flores200_dataset:
                #                 model_representations.append(last_token_activations)
                #             else:
                #                 model_representations.append(avg_activations)    

                #             ## now we are doing batch processing, so we need to print progress considering that it won't be divisible by 10.
                #             ## update the start index to indicate the number of samples processed, ensure that the start_index should always be lower than the total length of the dataset. Just added b
                #             start_index += len(out)
                #             # start_index += batch_size       ## this is not correct, much higher values.
                #             if len(generated_model_text) % printing_interval == 0:
                #                 print(f"Processed {len(generated_model_text)}/{len(all_inputs)} samples. Start_index: {start_index}")

                #     except RuntimeError as e:
                #         if "out of memory" in str(e).lower():
                #             print(f"OOM error with batch_size={batch_size}, reducing batch size by half")
                #             torch.cuda.empty_cache()  # Clear the GPU memory
                #         else:
                #             raise e  # Re-raise other runtime errors

                # 1) build a true batched DataLoader
                # import ipdb; ipdb.set_trace()
                batch_size = args.batch_size
                loader = DataLoader(MyDataset(all_inputs),batch_size=batch_size, collate_fn=collate_fn,drop_last=False,shuffle=False)
                model_representations = []
                which_token = None
                with torch.no_grad():
                    for batch in tqdm(loader):
                        input_ids = batch["input_ids"]  # [B, S]
                        # boolean mask of valid (non‐pad) tokens
                        attention_mask = input_ids.ne(tokenizer.pad_token_id)  # [B, S]
                        # assert (attention_mask == batch["attention_mask"]).all()

                        # 2) forward through model directly
                        outputs = model(input_ids=input_ids,attention_mask=attention_mask,output_hidden_states=True,return_dict_in_generate=True)
                        # outputs.hidden_states is a tuple of length num_layers+1
                        hs = torch.stack(outputs.hidden_states, dim=0)       # [L, B, S, D]

                        # 3) per‐layer average over non‐pad tokens
                        #    build mask of shape [L, B, S, 1]
                        m = attention_mask.float().unsqueeze(0).unsqueeze(-1)  # [1, B, S, 1]
                        m = m.expand_as(hs)                                    # [L, B, S, D]
                        
                        if args.use_flores200_dataset or args.use_wildguard_es or args.use_wildguard_de or args.use_wildguard_fr or args.use_wildguard_ko or args.use_wildguard_ru or args.use_wildguard_zh or args.use_wildguard_it or args.use_wildguard_pt or args.use_wildguard_nl or args.use_wildguard_caesar or args.use_wildguard_caesar1 or args.use_wildguard_leet or args.use_wildguard_vowel or args.use_wildguard_base64 or args.use_wildguard_hexadecimal or args.use_wildguard_alphanumeric or args.use_wildguard_ascii or args.use_wildguard_leet or args.last_token_representations:
                            which_token = "last"
                            ## assert last two tokens are not pad tokens
                            assert (input_ids[:, -1] != tokenizer.pad_token_id).all() and (attention_mask[:, -1] == 1).all()
                            # per‐layer last‐token activations
                            last_h    = hs[:, :, -1, :].clone()               # [L, B, D]
                            batch_repr = last_h.permute(1, 0, 2).cpu()      # [B, L, D]
                        elif args.avg_token_representations:
                            which_token = "avg"
                            # per‐layer average over non‐pad tokens
                            summed   = (hs * m).sum(dim=2)    # [L, B, D]
                            counts   = m.sum(dim=2)          # [L, B, D]
                            avg_repr = summed / counts       # [L, B, D]
                            batch_repr = avg_repr.permute(1, 0, 2).cpu()  # [B, L, D]
                        else:
                            raise NotImplementedError
                        
                        model_representations.append(batch_repr)  # [B, L, D]
                        ## print every 10th batch
                        if len(model_representations) % 10 == 0:
                            print(f"Processed {which_token} samples: {len(model_representations)*batch_size} of {len(all_inputs)}. Start_index: {start_index}")

                # 4) stitch all batches, save
                model_representations = torch.cat(model_representations, dim=0)        # [N, L, D]
                # torch.save(model_reprs, output_representation_file)

            print(f"Time taken: {time.time() - start_time}")
            # print(f"Length of generated_model_text: {len(generated_model_text)}")
            
            if args.llm_model not in ["multilingual-e5-large"]:
                # model_representations = torch.stack(model_representations)
                assert model_representations.shape == (len(all_inputs), model_num_layers, model_representations.shape[2])
            else:
                model_representations = torch.cat(model_representations)
                model_representations = model_representations.squeeze()
                assert model_representations.shape == (len(all_inputs), model_representations.shape[1]), f"Shape of the model representations is not correct: {model_representations.shape}"
            
            # import ipdb; ipdb.set_trace()
            print(f"saving results to {output_representation_file}")
            torch.save(model_representations, output_representation_file)

            if args.use_multijail_data or args.use_benign_data:     ## do not need with toxigen data
                with open(generated_output_text_file, "w") as f:
                    f.write(f"Time taken: {time.time() - start_time}")
                    for output in generated_model_text:
                        output = output.replace("\n", " ").replace("\r", " ")
                        f.write(output + "\n")
            print("saved the representations and generated text to the files.")
            
        else:
            raise NotImplementedError


def convert_english_to_cipher_languages(args):
    ## load the english dataset and convert it to the cipher languages and store it. 
    if args.use_toxigen_data:
        english_sentences = pd.read_csv("difficult_sampled_toxigen/difficult_sampled_toxigen_en.csv", sep="|")     ## inputs|jailbreak_label
        assert english_sentences.shape == (1000, 2)
        assert (english_sentences['jailbreak_label'][:500] == 1).all() and (english_sentences['jailbreak_label'][500:] == 0).all()
    
    elif args.use_random_sampled_toxigen_benign:
        english_sentences_benign = "toxigen_dataset/benign_en.txt"
        english_sentences_jailbreak = "toxigen_dataset/harmful_en.txt"
        ## read these txt files into two lists and then combine into one list
        with open(english_sentences_benign, "r") as f:
            benign_sentences = f.readlines()
        with open(english_sentences_jailbreak, "r") as f:
            jailbreak_sentences = f.readlines()
        assert len(benign_sentences) == len(jailbreak_sentences) == 1000
        
    elif args.use_length_filtered_aegis_benign:
        english_sentence_benign = "aegis_safety_dataset/benign_en.txt"
        english_sentence_jailbreak = "aegis_safety_dataset/harmful_en.txt"
        ## read these txt files into two lists and then combine into one list
        with open(english_sentence_benign, "r") as f:
            benign_sentences = f.readlines()
        with open(english_sentence_jailbreak, "r") as f:
            jailbreak_sentences = f.readlines()
    
    elif args.use_jailbreakbench_dataset_benign:
        english_sentence_benign = "jailbreakbench_dataset/benign_en.txt"
        english_sentence_jailbreak = "jailbreakbench_dataset/harmful_en.txt"
        ## read these txt files into two lists and then combine into one list
        with open(english_sentence_benign, "r") as f:
            benign_sentences = f.readlines()
        with open(english_sentence_jailbreak, "r") as f:
            jailbreak_sentences = f.readlines()
        assert len(benign_sentences) == len(jailbreak_sentences) == 100
        
    elif args.use_xstest_dataset_benign:
        english_sentence_benign = "xstest_dataset/benign_en.txt"
        english_sentence_jailbreak = "xstest_dataset/harmful_en.txt"
        ## read these txt files into two lists and then combine into one list
        with open(english_sentence_benign, "r") as f:
            benign_sentences = f.readlines()
        with open(english_sentence_jailbreak, "r") as f:
            jailbreak_sentences = f.readlines()
        assert len(benign_sentences) == len(jailbreak_sentences) == 200
    
    elif args.use_beavertails_rlhf_dataset_benign:
        english_sentence_benign = "beavertails_rlhf_dataset/benign_en.txt"
        english_sentence_jailbreak = "beavertails_rlhf_dataset/harmful_en.txt"
        ## read these txt files into two lists and then combine into one list
        with open(english_sentence_benign, "r") as f:
            benign_sentences = f.readlines()
        with open(english_sentence_jailbreak, "r") as f:
            jailbreak_sentences = f.readlines()
        assert len(benign_sentences) == len(jailbreak_sentences) == 2000
    
    elif args.use_sst2_dataset:
        english_sentence_benign = "sentiment_classification/positive_phrases.txt"
        english_sentence_jailbreak = "sentiment_classification/negative_phrases.txt"
        ## read these txt files into two lists and then combine into one list
        with open(english_sentence_benign, "r") as f:
            benign_sentences = f.readlines()
        with open(english_sentence_jailbreak, "r") as f:
            jailbreak_sentences = f.readlines()
        assert len(benign_sentences) == len(jailbreak_sentences) == 500
    
    elif args.use_flores200_dataset:
        english_sentence_benign = "flores200_dataset/benign_en.txt"
        with open(english_sentence_benign, "r") as f:
            benign_sentences = f.readlines()
        jailbreak_sentences = []
        assert len(benign_sentences) == 2009
    
    elif args.use_wildguard_ciphers:
        english_sentence_benign_file = f"wildguard_datasets/mWildGuardMix-train-tower-nllb-v2.1/wildguard-train-English.parquet"
        english_sentences_benign_content = pd.read_parquet(english_sentence_benign_file)
        benign_sentences = english_sentences_benign_content['prompt'].tolist()
        assert len(benign_sentences) == 86759, f"Length of benign sentences: {len(benign_sentences)}"
        jailbreak_sentences = []
    
    elif args.use_wikitext_sampled:
        english_sentence_benign = "wikitext_sampled/benign_en.txt"
        with open(english_sentence_benign, "r") as f:
            benign_sentences = f.readlines()
        jailbreak_sentences = []
        assert len(benign_sentences) == 1000
    
    elif args.use_llm_lat_harmful:
        english_sentence_benign = "llm_lat_dataset/all_final_prompts/harmful_en.txt"
        with open(english_sentence_benign, "r") as f:
            benign_sentences = f.readlines()
        jailbreak_sentences = []
        assert len(benign_sentences) == 10999

    elif args.use_llm_lat_benign:
        english_sentence_benign = "llm_lat_dataset/all_final_prompts/benign_en.txt"
        with open(english_sentence_benign, "r") as f:
            benign_sentences = f.readlines()
        jailbreak_sentences = []
        assert len(benign_sentences) == 10982
    
    elif args.use_smaller_aegis_resampled_benign:
        english_sentence_benign = "second_time_aegis_safety_dataset/benign_en.txt"
        english_sentence_jailbreak = "second_time_aegis_safety_dataset/harmful_en.txt"
        with open(english_sentence_benign, "r") as f:
            benign_sentences = f.readlines()
        with open(english_sentence_jailbreak, "r") as f:
            jailbreak_sentences = f.readlines()
        assert len(benign_sentences) == 1426, f"Length of benign sentences: {len(benign_sentences)}"
        assert len(jailbreak_sentences) == 1432, f"Length of jailbreak sentences: {len(jailbreak_sentences)}"
    
    elif args.use_harmbench:
        english_sentence_jailbreak = "harmbench_dataset/harmful_en.txt"
        with open(english_sentence_jailbreak, "r") as f:
            jailbreak_sentences = f.readlines()
        benign_sentences = []
        assert len(jailbreak_sentences) == 224, f"Length of jailbreak sentences: {len(jailbreak_sentences)}"
    
    elif args.use_advbench_dataset:
        english_sentence_jailbreak = "advbench_dataset/harmful_en.txt"
        with open(english_sentence_jailbreak, "r") as f:
            jailbreak_sentences = f.readlines()
        benign_sentences = []
        assert len(jailbreak_sentences) == 520, f"Length of jailbreak sentences: {len(jailbreak_sentences)}"
        
    elif args.use_forbidden_questions:
        english_sentence_jailbreak = "forbidden_questions_dataset/harmful_en.txt"
        with open(english_sentence_jailbreak, "r") as f:
            jailbreak_sentences = f.readlines()
        benign_sentences = []
        assert len(jailbreak_sentences) == 240, f"Length of jailbreak sentences: {len(jailbreak_sentences)}"
    
    elif args.use_saladbench_dataset:
        english_sentence_jailbreak = "saladbench_dataset/harmful_en.txt"
        with open(english_sentence_jailbreak, "r") as f:
            jailbreak_sentences = f.readlines()
        benign_sentences = []
        assert len(jailbreak_sentences) == 1001, f"Length of jailbreak sentences: {len(jailbreak_sentences)}"
        
    elif args.use_simple_safety_tests_dataset:
        english_sentence_jailbreak = "simple_safety_tests_dataset/harmful_en.txt"
        with open(english_sentence_jailbreak, "r") as f:
            jailbreak_sentences = f.readlines()
        benign_sentences = []
        assert len(jailbreak_sentences) == 100, f"Length of jailbreak sentences: {len(jailbreak_sentences)}"
        
    elif args.use_filtered_toxicchat_benign:
        english_sentence_benign = "toxicchat_dataset/benign_en.txt"
        english_sentence_jailbreak = "toxicchat_dataset/harmful_en.txt"
        with open(english_sentence_benign, "r") as f:
            benign_sentences = f.readlines()
        with open(english_sentence_jailbreak, "r") as f:
            jailbreak_sentences = f.readlines()
        assert len(benign_sentences) == 250, f"Length of benign sentences: {len(benign_sentences)}"
        assert len(jailbreak_sentences) == 247, f"Length of jailbreak sentences: {len(jailbreak_sentences)}"
        
    elif args.use_toxicity_jigsaw_dataset_benign:
        english_sentence_benign = "toxicity_jigsaw_dataset/benign_en.txt"
        english_sentence_jailbreak = "toxicity_jigsaw_dataset/harmful_en.txt"
        with open(english_sentence_benign, "r") as f:
            benign_sentences = f.readlines()
        with open(english_sentence_jailbreak, "r") as f:
            jailbreak_sentences = f.readlines()
        assert len(benign_sentences) == 658, f"Length of benign sentences: {len(benign_sentences)}"
        assert len(jailbreak_sentences) == 689, f"Length of jailbreak sentences: {len(jailbreak_sentences)}"
        
    elif args.use_toxic_text_dataset_benign:
        english_sentence_benign = "toxic_text_dataset/benign_en.txt"
        english_sentence_jailbreak = "toxic_text_dataset/harmful_en.txt"
        with open(english_sentence_benign, "r") as f:
            benign_sentences = f.readlines()
        with open(english_sentence_jailbreak, "r") as f:
            jailbreak_sentences = f.readlines()
        assert len(benign_sentences) == 997, f"Length of benign sentences: {len(benign_sentences)}"
        assert len(jailbreak_sentences) == 998, f"Length of jailbreak sentences: {len(jailbreak_sentences)}"
    
    else:
        raise NotImplementedError

    # cipher_languages = ['hexadecimal', 'base64', 'leet', 'vowel', 'alphanumeric', 'atbash', 'morse', 'caesar1', 'caesar2', 'caesar', 'caesar4', 'caesar5', 'caesar6', 'caesar7', 'caesar8', 'caesar9', 'caesarneg1', 'caesarneg2', 'caesarneg3', 'caesarneg4', 'caesarneg5', 'caesarneg6', 'caesarneg7', 'caesarneg8', 'caesarneg9', 'ascii',]
    cipher_languages = ['caesar1', 'caesar2', 'caesar', 'caesar4', 'caesar5', 'caesar6', 'caesar7', 'caesarneg1', 'caesarneg2', 'caesarneg3', 'caesarneg4', 'caesarneg5', 'caesarneg6', 'caesarneg7', 'ascii', 'hexadecimal', 'base64', 'leet', 'vowel', 'alphanumeric']
    
    from encode_experts import encode_expert_dict
    
    for lang_cipher in cipher_languages:
        print(f"Converting to language: {lang_cipher}")
        translated_sentences = []
        if args.use_toxigen_data:
            for sentence in english_sentences['inputs']:
                translated_sentence = encode_expert_dict[lang_cipher].encode(sentence)
                translated_sentences.append(translated_sentence)
        
            assert len(translated_sentences) == 1000
            translated_sentences_df = pd.DataFrame({'inputs': translated_sentences, 'jailbreak_label': english_sentences['jailbreak_label']})
            assert translated_sentences_df.shape == (1000, 2)
            if lang_cipher == "leet":
                translated_sentences_df.to_csv("difficult_sampled_toxigen/difficult_sampled_toxigen_leet.csv", sep='Ʃ', index=False)
            else:
                translated_sentences_df.to_csv(f"difficult_sampled_toxigen/difficult_sampled_toxigen_{lang_cipher}.csv", sep="|", index=False)
        
        elif args.use_random_sampled_toxigen_benign or args.use_length_filtered_aegis_benign or args.use_jailbreakbench_dataset_benign or args.use_xstest_dataset_benign or args.use_beavertails_rlhf_dataset_benign or args.use_sst2_dataset or args.use_flores200_dataset or args.use_wildguard_ciphers or args.use_wikitext_sampled or args.use_llm_lat_harmful or args.use_llm_lat_benign or args.use_smaller_aegis_resampled_benign or args.use_harmbench or args.use_advbench_dataset or args.use_forbidden_questions or args.use_saladbench_dataset or args.use_simple_safety_tests_dataset or args.use_filtered_toxicchat_benign or args.use_toxicity_jigsaw_dataset_benign or args.use_toxic_text_dataset_benign:
            translated_benign_sentences = []
            translated_jailbreak_sentences = []
            for sentence in benign_sentences:
                translated_benign_sentences.append(encode_expert_dict[lang_cipher].encode(sentence))
            
            for sentence in jailbreak_sentences:
                translated_jailbreak_sentences.append(encode_expert_dict[lang_cipher].encode(sentence))
            
            if args.use_random_sampled_toxigen_benign:
                assert len(translated_benign_sentences) == len(translated_jailbreak_sentences) == 1000
            
            if lang_cipher in ["unicode", "morse", "utf", "ascii"]:
                ## each of the sentences have "\n\n" at the end, replace it with "\n"
                translated_benign_sentences = [sentence.replace("\n\n", "\n") for sentence in translated_benign_sentences]
                translated_jailbreak_sentences = [sentence.replace("\n\n", "\n") for sentence in translated_jailbreak_sentences]
            elif lang_cipher in ["hexadecimal", "base64", "binary", "pigpen"]:
                ## none of these sentences have "\n" at the end, add it if it is not there.
                translated_benign_sentences = [sentence + "\n" if len(sentence) > 0 and sentence[-1] != "\n" else sentence for sentence in translated_benign_sentences]
                translated_jailbreak_sentences = [sentence + "\n" if len(sentence) > 0 and sentence[-1] != "\n" else sentence for sentence in translated_jailbreak_sentences]
            elif lang_cipher in ["reverse"]:
                ## all these sentences have "\n" at the start, remove it. and add it at the end if it is not there.
                translated_benign_sentences = [sentence[1:] if sentence[0] == "\n" else sentence for sentence in translated_benign_sentences]
                translated_jailbreak_sentences = [sentence[1:] if sentence[0] == "\n" else sentence for sentence in translated_jailbreak_sentences]
                translated_benign_sentences = [sentence + "\n" if sentence[-1] != "\n" else sentence for sentence in translated_benign_sentences]
                translated_jailbreak_sentences = [sentence + "\n" if sentence[-1] != "\n" else sentence for sentence in translated_jailbreak_sentences]
            else:
                pass
            
            if args.use_random_sampled_toxigen_benign:
                with open(f"toxigen_dataset/benign_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_benign_sentences:
                        f.write(sentence)
                with open(f"toxigen_dataset/harmful_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_jailbreak_sentences:
                        f.write(sentence)
                        
                ## read the stored files and asserr they have 500 lines each
                with open(f"toxigen_dataset/benign_{lang_cipher}.txt", "r") as f:
                    temp_benign_sentences = f.readlines()
                with open(f"toxigen_dataset/harmful_{lang_cipher}.txt", "r") as f:
                    temp_jailbreak_sentences = f.readlines()
                    
                assert len(temp_benign_sentences) == len(temp_jailbreak_sentences) == 1000, f"Lengths: {len(temp_benign_sentences)} and {len(temp_jailbreak_sentences)}"
        
            elif args.use_length_filtered_aegis_benign:
                with open(f"aegis_safety_dataset/benign_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_benign_sentences:
                        f.write(sentence)
                with open(f"aegis_safety_dataset/harmful_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_jailbreak_sentences:
                        f.write(sentence)

                ## read the stored files and asserr they have either 1409 or 1410 lines each
                with open(f"aegis_safety_dataset/benign_{lang_cipher}.txt", "r") as f:
                    temp_benign_sentences = f.readlines()
                with open(f"aegis_safety_dataset/harmful_{lang_cipher}.txt", "r") as f:
                    temp_jailbreak_sentences = f.readlines()
            
                assert len(temp_benign_sentences) == 1409 and len(temp_jailbreak_sentences) == 1410, f"Lengths: {len(temp_benign_sentences)} and {len(temp_jailbreak_sentences)}"
                
            elif args.use_jailbreakbench_dataset_benign:
                with open(f"jailbreakbench_dataset/benign_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_benign_sentences:
                        f.write(sentence)
                with open(f"jailbreakbench_dataset/harmful_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_jailbreak_sentences:
                        f.write(sentence)
                
                ## read the stored files and asserr they have 500 lines each
                with open(f"jailbreakbench_dataset/benign_{lang_cipher}.txt", "r") as f:
                    temp_benign_sentences = f.readlines()
                with open(f"jailbreakbench_dataset/harmful_{lang_cipher}.txt", "r") as f:
                    temp_jailbreak_sentences = f.readlines()
                    
                assert len(temp_benign_sentences) == len(temp_jailbreak_sentences) == 100, f"Lengths: {len(temp_benign_sentences)} and {len(temp_jailbreak_sentences)}"
                
            elif args.use_xstest_dataset_benign:
                with open(f"xstest_dataset/benign_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_benign_sentences:
                        f.write(sentence)
                with open(f"xstest_dataset/harmful_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_jailbreak_sentences:
                        f.write(sentence)
                
                ## read the stored files and asserr they have 100 lines each
                with open(f"xstest_dataset/benign_{lang_cipher}.txt", "r") as f:
                    temp_benign_sentences = f.readlines()
                with open(f"xstest_dataset/harmful_{lang_cipher}.txt", "r") as f:
                    temp_jailbreak_sentences = f.readlines()
                    
                assert len(temp_benign_sentences) == len(temp_jailbreak_sentences) == 200, f"Lengths: {len(temp_benign_sentences)} and {len(temp_jailbreak_sentences)}"
            
            elif args.use_beavertails_rlhf_dataset_benign:
                with open(f"beavertails_rlhf_dataset/benign_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_benign_sentences:
                        f.write(sentence)
                with open(f"beavertails_rlhf_dataset/harmful_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_jailbreak_sentences:
                        f.write(sentence)
                
                ## read the stored files and asserr they have 1000 lines each
                with open(f"beavertails_rlhf_dataset/benign_{lang_cipher}.txt", "r") as f:
                    temp_benign_sentences = f.readlines()
                with open(f"beavertails_rlhf_dataset/harmful_{lang_cipher}.txt", "r") as f:
                    temp_jailbreak_sentences = f.readlines()
                    
                assert len(temp_benign_sentences) == len(temp_jailbreak_sentences) == 2000, f"Lengths: {len(temp_benign_sentences)} and {len(temp_jailbreak_sentences)}"
            
            elif args.use_sst2_dataset:
                with open(f"sentiment_classification/translated_sst2_dataset/positive_phrases_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_benign_sentences:
                        f.write(sentence)
                with open(f"sentiment_classification/translated_sst2_dataset/negative_phrases_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_jailbreak_sentences:
                        f.write(sentence)
                
                ## read the stored files and asserr they have 500 lines each
                with open(f"sentiment_classification/translated_sst2_dataset/positive_phrases_{lang_cipher}.txt", "r") as f:
                    temp_benign_sentences = f.readlines()
                with open(f"sentiment_classification/translated_sst2_dataset/negative_phrases_{lang_cipher}.txt", "r") as f:
                    temp_jailbreak_sentences = f.readlines()
                    
                assert len(temp_benign_sentences) == len(temp_jailbreak_sentences) == 500, f"Lengths: {len(temp_benign_sentences)} and {len(temp_jailbreak_sentences)}"
            
            elif args.use_flores200_dataset:
                with open(f"flores200_dataset/benign_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_benign_sentences:
                        f.write(sentence)
                
                ## read the stored files and asserr they have 500 lines each
                with open(f"flores200_dataset/benign_{lang_cipher}.txt", "r") as f:
                    temp_benign_sentences = f.readlines()
                    
                assert len(temp_benign_sentences) == 2009, f"Lengths: {len(temp_benign_sentences)}"
            
            elif args.use_wildguard_ciphers:
                ## write it in a parquet file. Keep all other columns same as english_sentences_benign_content, replace the prompt column with the translated sentences.
                english_sentences_benign_content['prompt'] = translated_benign_sentences
                english_sentences_benign_content.to_parquet(f"wildguard_datasets/mWildGuardMix-train-tower-nllb-v2.1/wildguard-train-{lang_cipher}.parquet")
                print(f"saved in wildguard_datasets/mWildGuardMix-train-tower-nllb-v2.1/wildguard-train-{lang_cipher}.parquet")
            
            elif args.use_wikitext_sampled:
                with open(f"wikitext_sampled/benign_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_benign_sentences:
                        f.write(sentence)
                
                ## read the stored files and asserr they have 1000 lines each
                with open(f"wikitext_sampled/benign_{lang_cipher}.txt", "r") as f:
                    temp_benign_sentences = f.readlines()
                    
                assert len(temp_benign_sentences) == 1000, f"Lengths: {len(temp_benign_sentences)}"
            
            elif args.use_llm_lat_harmful:
                with open(f"llm_lat_dataset/all_final_prompts/harmful_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_benign_sentences:
                        f.write(sentence)
                
                ## read the stored files and asserr they have 1000 lines each
                with open(f"llm_lat_dataset/all_final_prompts/harmful_{lang_cipher}.txt", "r") as f:
                    temp_benign_sentences = f.readlines()
                    
                assert len(temp_benign_sentences) == 10999, f"Lengths: {len(temp_benign_sentences)}"
                
            elif args.use_llm_lat_benign:
                with open(f"llm_lat_dataset/all_final_prompts/benign_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_benign_sentences:
                        f.write(sentence)
                
                ## read the stored files and asserr they have 1000 lines each
                with open(f"llm_lat_dataset/all_final_prompts/benign_{lang_cipher}.txt", "r") as f:
                    temp_benign_sentences = f.readlines()
                    
                assert len(temp_benign_sentences) == 10982, f"Lengths: {len(temp_benign_sentences)}"
            
            elif args.use_smaller_aegis_resampled_benign:
                with open(f"second_time_aegis_safety_dataset/benign_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_benign_sentences:
                        f.write(sentence)
                with open(f"second_time_aegis_safety_dataset/harmful_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_jailbreak_sentences:
                        f.write(sentence)
                
                ## read the stored files and asserr they have 1000 lines each
                with open(f"second_time_aegis_safety_dataset/benign_{lang_cipher}.txt", "r") as f:
                    temp_benign_sentences = f.readlines()
                with open(f"second_time_aegis_safety_dataset/harmful_{lang_cipher}.txt", "r") as f:
                    temp_jailbreak_sentences = f.readlines()
                    
                assert len(temp_benign_sentences) == 1426 and len(temp_jailbreak_sentences) == 1432, f"Lengths: {len(temp_benign_sentences)} and {len(temp_jailbreak_sentences)}"
            
            elif args.use_harmbench:
                with open(f"harmbench_dataset/harmful_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_jailbreak_sentences:
                        f.write(sentence)
                
                ## read the stored files and asserr they have 1000 lines each
                with open(f"harmbench_dataset/harmful_{lang_cipher}.txt", "r") as f:
                    temp_jailbreak_sentences = f.readlines()
                    
                assert len(temp_jailbreak_sentences) == 224, f"Lengths: {len(temp_jailbreak_sentences)}"
            
            elif args.use_advbench_dataset:
                with open(f"advbench_dataset/harmful_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_jailbreak_sentences:
                        f.write(sentence)
                
                ## read the stored files and asserr they have 1000 lines each
                with open(f"advbench_dataset/harmful_{lang_cipher}.txt", "r") as f:
                    temp_jailbreak_sentences = f.readlines()
                    
                assert len(temp_jailbreak_sentences) == 520, f"Lengths: {len(temp_jailbreak_sentences)}"
            
            elif args.use_forbidden_questions:
                with open(f"forbidden_questions_dataset/harmful_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_jailbreak_sentences:
                        f.write(sentence)
                
                ## read the stored files and asserr they have 1000 lines each
                with open(f"forbidden_questions_dataset/harmful_{lang_cipher}.txt", "r") as f:
                    temp_jailbreak_sentences = f.readlines()
                    
                assert len(temp_jailbreak_sentences) == 240, f"Lengths: {len(temp_jailbreak_sentences)}"
        
            elif args.use_saladbench_dataset:
                with open(f"saladbench_dataset/harmful_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_jailbreak_sentences:
                        f.write(sentence)
                
                ## read the stored files and asserr they have 1000 lines each
                with open(f"saladbench_dataset/harmful_{lang_cipher}.txt", "r") as f:
                    temp_jailbreak_sentences = f.readlines()
                    
                assert len(temp_jailbreak_sentences) == 1001, f"Lengths: {len(temp_jailbreak_sentences)}"
            
            elif args.use_simple_safety_tests_dataset:
                with open(f"simple_safety_tests_dataset/harmful_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_jailbreak_sentences:
                        f.write(sentence)
                
                ## read the stored files and asserr they have 1000 lines each
                with open(f"simple_safety_tests_dataset/harmful_{lang_cipher}.txt", "r") as f:
                    temp_jailbreak_sentences = f.readlines()
                    
                assert len(temp_jailbreak_sentences) == 100, f"Lengths: {len(temp_jailbreak_sentences)}"
            
            elif args.use_filtered_toxicchat_benign:
                with open(f"toxicchat_dataset/benign_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_benign_sentences:
                        f.write(sentence)
                with open(f"toxicchat_dataset/harmful_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_jailbreak_sentences:
                        f.write(sentence)
                
                ## read the stored files and asserr they have 1000 lines each
                with open(f"toxicchat_dataset/benign_{lang_cipher}.txt", "r") as f:
                    temp_benign_sentences = f.readlines()
                with open(f"toxicchat_dataset/harmful_{lang_cipher}.txt", "r") as f:
                    temp_jailbreak_sentences = f.readlines()
                    
                assert len(temp_benign_sentences) == 250 and len(temp_jailbreak_sentences) == 247, f"Lengths: {len(temp_benign_sentences)} and {len(temp_jailbreak_sentences)}"
            
            elif args.use_toxicity_jigsaw_dataset_benign:
                with open(f"toxicity_jigsaw_dataset/benign_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_benign_sentences:
                        f.write(sentence)
                with open(f"toxicity_jigsaw_dataset/harmful_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_jailbreak_sentences:
                        f.write(sentence)
                
                ## read the stored files and asserr they have 1000 lines each
                with open(f"toxicity_jigsaw_dataset/benign_{lang_cipher}.txt", "r") as f:
                    temp_benign_sentences = f.readlines()
                with open(f"toxicity_jigsaw_dataset/harmful_{lang_cipher}.txt", "r") as f:
                    temp_jailbreak_sentences = f.readlines()
                    
                assert len(temp_benign_sentences) == 658 and len(temp_jailbreak_sentences) == 689, f"Lengths: {len(temp_benign_sentences)} and {len(temp_jailbreak_sentences)}"
            
            elif args.use_toxic_text_dataset_benign:
                with open(f"toxic_text_dataset/benign_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_benign_sentences:
                        f.write(sentence)
                with open(f"toxic_text_dataset/harmful_{lang_cipher}.txt", "w") as f:
                    for sentence in translated_jailbreak_sentences:
                        f.write(sentence)
                
                ## read the stored files and asserr they have 1000 lines each
                with open(f"toxic_text_dataset/benign_{lang_cipher}.txt", "r") as f:
                    temp_benign_sentences = f.readlines()
                with open(f"toxic_text_dataset/harmful_{lang_cipher}.txt", "r") as f:
                    temp_jailbreak_sentences = f.readlines()
                    
                assert len(temp_benign_sentences) == 997 and len(temp_jailbreak_sentences) == 998, f"Lengths: {len(temp_benign_sentences)} and {len(temp_jailbreak_sentences)}"
            
            else:
                raise NotImplementedError
        
        else:
            raise NotImplementedError
        
        print(f"Done with language: {lang_cipher}")
            
    print(f"Done with all {len(cipher_languages)} languages.")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Get difference in model vulnerabilities across languages.')
    parser.add_argument('--produce_responses', action='store_true', help='produce responses from the model for the jailbreaking prompts')
    parser.add_argument('--llm_model', type=str, choices=['gpt-4-azure', 'gpt-4o', 'adversarial-gpt', "llama3.1-8b", "llama3.1-8b-instruct", "llama3.1-70b", "llama3.1-70b-instruct", "llama3.3-70b-instruct", "mistral-nemo-12b", "multilingual-e5-large", "qwen-2.5-72B-instruct","qwen2.5-32b-instruct", "qwen2.5-0.5b-instruct"], help='choose the model to use for generating responses', default=None)
    # parser.add_argument('--aim_prompt', action='store_true', help='use AIM prompt for the jailbreaking prompts')
    parser.add_argument('--judge_responses', action='store_true', help='judge the responses from the model for the jailbreaking prompts')
    parser.add_argument('--parallelize_query_response', action='store_true', help='parallelize the responses from the model')
    parser.add_argument('--test', action='store_true', help='test the model')
    parser.add_argument('--parse_judgements', action='store_true', help='parse the judgements from the model')
    parser.add_argument('--inspect_judgements', action='store_true', help='inspect the judgements for the unsafe responses')
    parser.add_argument('--separate_query_and_responses', action='store_true', help='separate the query and responses in two separate csv files')
    # parser.add_argument('--separate_responses_and_judgements', action='store_true', help='separate the responses and judgements in two separate csv files')
    parser.add_argument('--jailbreak_prompt', choices=['aim', 'cipher', 'aligned', 'code_nesting', 'moralizing_rant', 'gpt4real', 'refusal_suppression', 'table_nesting', 'burple', 'switch', 'mr_blonde', 'complex', 'balakula', 'text_continuation_nesting', 'jedi_mind_trick', 'violet', ], help='choose the jailbreaking prompt to use', type=str)
    
    ## Settings for the classifier training. 
    parser.add_argument('--get_hidden_layer_representations', action='store_true', help='get the hidden layer representations for the responses')
    parser.add_argument('--for_hidden_layer_representations_language_index', type=int, default=0, help='get the hidden layer representations for the responses')
    parser.add_argument('--offload_specific_layer_representations', type=int, default=None, help='offload the specific layer representations for the responses')
    parser.add_argument('--last_token_representations', action='store_true', help='get the last token representations for the responses')
    parser.add_argument('--avg_token_representations', action='store_true', help='get the average token representations for the responses')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for the model')
    parser.add_argument('--add_no_translator_command_english', action='store_true', help='add the no translator command to the queries for non-english languages')
    parser.add_argument('--add_no_translator_command_respective_language', action='store_true', help='add the no translator command to the queries in non-english languages in that language itself')
    parser.add_argument('--add_internally_translate_but_output_same_language', action='store_true', help='add the internally translate but output same language command to the queries for non-english languages')
    parser.add_argument('--train_classifiers_on_representations', action='store_true', help='train classifiers on the representations to separate benign and jailbreaking prompts')
    parser.add_argument('--classifier_arch', type=str, default=None, choices=['linear', 'ridge', 'logistic'], help='choose the classifier type to use for training the classifier')
    
    parser.add_argument('--get_benign_dataset', action='store_true', help='get the benign dataset')
    parser.add_argument('--use_benign_data', action='store_true', help='use the benign data for getting representations instead of the multijail data')
    parser.add_argument('--use_multijail_data', action='store_true', help='use the multijail data for getting representations instead of the benign data')
    parser.add_argument("--use_toxigen_data", action='store_true', help='use the toxigen data for getting representations instead of the benign data')
    parser.add_argument("--use_random_sampled_toxigen_benign", action='store_true', help='use the random sampled toxigen data for getting representations instead of the benign data')
    parser.add_argument("--use_random_sampled_toxigen_harmful", action='store_true', help='use the random sampled toxigen data for getting representations instead of the benign data')
    parser.add_argument("--use_length_filtered_aegis_benign", action='store_true', help='use the length filtered aegis benign data for getting representations instead of the benign data')
    parser.add_argument("--use_length_filtered_aegis_harmful", action='store_true', help='use the length filtered aegis harmful data for getting representations instead of the benign data')
    parser.add_argument("--use_jailbreakbench_dataset_benign", action='store_true', help='use the jailbreakbench dataset for getting representations instead of the benign data')
    parser.add_argument("--use_jailbreakbench_dataset_harmful", action='store_true', help='use the jailbreakbench dataset for getting representations instead of the benign data')
    parser.add_argument("--use_xstest_dataset_benign", action='store_true', help='use the xstest dataset for getting representations instead of the benign data')
    parser.add_argument("--use_xstest_dataset_harmful", action='store_true', help='use the xstest dataset for getting representations instead of the benign data')
    parser.add_argument("--use_beavertails_rlhf_dataset_benign", action='store_true', help='use the beavertails_rlhf dataset for getting representations instead of the benign data')
    parser.add_argument("--use_beavertails_rlhf_dataset_harmful", action='store_true', help='use the beavertails_rlhf dataset for getting representations instead of the benign data')
    parser.add_argument("--use_flores200_dataset", action='store_true', help='use the flores200 dataset for getting representations instead of the benign data')
    
    parser.add_argument("--use_polyguardmix_train_100K_benign", action='store_true', help='use the polyguardmix dataset for getting representations instead of the benign data')
    parser.add_argument("--use_polyguardmix_train_100K_harmful", action='store_true', help='use the polyguardmix dataset for getting representations instead of the benign data')
    parser.add_argument("--use_polyguardmix_train_500K_benign", action='store_true', help='use the polyguardmix dataset for getting representations instead of the benign data')
    parser.add_argument("--use_polyguardmix_train_500K_harmful", action='store_true', help='use the polyguardmix dataset for getting representations instead of the benign data')
    parser.add_argument("--use_polyguardmix_train_all_benign", action='store_true', help='use the polyguardmix dataset for getting representations instead of the benign data')
    parser.add_argument("--polyguardmix_train_all_chunk", type=int, default=None, help='which chunk of the polyguardmix dataset to use for getting representations instead of the benign data')
    parser.add_argument("--use_polyguardmix_train_all_harmful", action='store_true', help='use the polyguardmix dataset for getting representations instead of the benign data')
    parser.add_argument("--use_polyguardmix_test_benign_prompt", action='store_true', help='use the polyguardmix dataset for getting representations instead of the benign data')
    parser.add_argument("--use_polyguardmix_test_harmful_prompt", action='store_true', help='use the polyguardmix dataset for getting representations instead of the benign data')
    parser.add_argument("--use_polyguardmix_test_benign_response", action='store_true', help='use the polyguardmix dataset for getting representations instead of the benign data')
    parser.add_argument("--use_polyguardmix_test_harmful_response", action='store_true', help='use the polyguardmix dataset for getting representations instead of the benign data')
    
    parser.add_argument("--use_wildguard_ar", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_zh", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_cs", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_nl", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_en", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_fr", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_de", action='store_true', help='use the wildguard-en dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_hi", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_it", action='store_true', help='use the wildguard-it dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_ja", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_ko", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_po", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_pt", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_ru", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_es", action='store_true', help='use the wildguard-es dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_sv", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_th", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')

    parser.add_argument("--use_wildguard_ciphers", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_caesar", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_caesar1", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_leet", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_vowel", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_base64", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_hexadecimal", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_alphanumeric", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_ascii", action='store_true', help='use the wildguard-fr dataset for getting representations instead of the benign data')
    
    parser.add_argument("--use_wildguardmix_sampled_benign", action='store_true', help='use the wildguardmix-sampled dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguardmix_sampled_harmful", action='store_true', help='use the wildguardmix-sampled dataset for getting representations instead of the benign data')
    
    parser.add_argument("--use_wikidata_en", action='store_true', help='use the wikidata-en dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wikidata_fr", action='store_true', help='use the wikidata-en dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wikitext_sampled", action='store_true', help='use the wikitext-sampled dataset for getting representations instead of the benign data')
    
    parser.add_argument("--use_filtered_multijail_dataset_en", action='store_true', help='use the filtered multijail dataset for getting representations instead of the benign data')
    parser.add_argument("--use_filtered_multijail_dataset_ar", action='store_true', help='use the filtered multijail dataset for getting representations instead of the benign data')
    parser.add_argument("--use_filtered_multijail_dataset_bn", action='store_true', help='use the filtered multijail dataset for getting representations instead of the benign data')
    parser.add_argument("--use_filtered_multijail_dataset_it", action='store_true', help='use the filtered multijail dataset for getting representations instead of the benign data')
    parser.add_argument("--use_filtered_multijail_dataset_jv", action='store_true', help='use the filtered multijail dataset for getting representations instead of the benign data')
    parser.add_argument("--use_filtered_multijail_dataset_ko", action='store_true', help='use the filtered multijail dataset for getting representations instead of the benign data')
    parser.add_argument("--use_filtered_multijail_dataset_sw", action='store_true', help='use the filtered multijail dataset for getting representations instead of the benign data')
    parser.add_argument("--use_filtered_multijail_dataset_th", action='store_true', help='use the filtered multijail dataset for getting representations instead of the benign data')
    parser.add_argument("--use_filtered_multijail_dataset_vi", action='store_true', help='use the filtered multijail dataset for getting representations instead of the benign data')
    parser.add_argument("--use_filtered_multijail_dataset_zh", action='store_true', help='use the filtered multijail dataset for getting representations instead of the benign data')
    parser.add_argument("--use_filtered_csrt_dataset", action='store_true', help='use the filtered csrt dataset for getting representations instead of the benign data')        ## this dataset has the same english prompts as the multijail dataset, hence same harmful indexes. 
    
    parser.add_argument("--use_xsafety_bn", action='store_true', help='use the xsafety-en dataset for getting representations instead of the benign data')
    parser.add_argument("--use_xsafety_fr", action='store_true', help='use the xsafety-en dataset for getting representations instead of the benign data')
    parser.add_argument("--use_xsafety_sp", action='store_true', help='use the xsafety-en dataset for getting representations instead of the benign data')
    parser.add_argument("--use_xsafety_zh", action='store_true', help='use the xsafety-en dataset for getting representations instead of the benign data')
    parser.add_argument("--use_xsafety_ar", action='store_true', help='use the xsafety-en dataset for getting representations instead of the benign data')
    parser.add_argument("--use_xsafety_hi", action='store_true', help='use the xsafety-en dataset for getting representations instead of the benign data')
    parser.add_argument("--use_xsafety_ja", action='store_true', help='use the xsafety-en dataset for getting representations instead of the benign data')
    parser.add_argument("--use_xsafety_ru", action='store_true', help='use the xsafety-en dataset for getting representations instead of the benign data')
    parser.add_argument("--use_xsafety_de", action='store_true', help='use the xsafety-en dataset for getting representations instead of the benign data')
    parser.add_argument("--use_xsafety_en", action='store_true', help='use the xsafety-en dataset for getting representations instead of the benign data')

    parser.add_argument("--use_rtplx_en", action='store_true', help='use the rtplx-en dataset for getting representations instead of the benign data')
    parser.add_argument("--use_rtplx_others", action='store_true', help='use the rtplx-en dataset for getting representations instead of the benign data')
    
    parser.add_argument("--use_llm_lat_harmful", action='store_true', help='use the llm-lat dataset for getting representations instead of the benign data')
    parser.add_argument("--use_llm_lat_benign", action='store_true', help='use the llm-lat dataset for getting representations instead of the benign data')
    parser.add_argument("--use_original_llm_lat_harmful", action='store_true', help='use the original llm-lat dataset for getting representations instead of the benign data')
    parser.add_argument("--use_original_llm_lat_benign", action='store_true', help='use the original llm-lat dataset for getting representations instead of the benign data')
    
    parser.add_argument("--use_aegis2_LG_filtered_benign", action='store_true', help='use the aegis2_LG_filtered dataset for getting representations instead of the benign data')
    parser.add_argument("--use_aegis2_LG_filtered_harmful", action='store_true', help='use the aegis2_LG_filtered dataset for getting representations instead of the benign data')
    parser.add_argument("--use_aegis2_random_sample_benign", action='store_true', help='use the aegis2_random_sample dataset for getting representations instead of the benign data')
    parser.add_argument("--use_aegis2_random_sample_harmful", action='store_true', help='use the aegis2_random_sample dataset for getting representations instead of the benign data')
    parser.add_argument("--use_third_time_aegis_resampled_benign", action='store_true', help='use the third_time_aegis_resampled dataset for getting representations instead of the benign data')
    parser.add_argument("--use_third_time_aegis_resampled_harmful", action='store_true', help='use the third_time_aegis_resampled dataset for getting representations instead of the benign data')
    parser.add_argument("--use_smaller_aegis_resampled_benign", action='store_true', help='use the smaller_aegis_resampled dataset for getting representations instead of the benign data')
    parser.add_argument("--use_smaller_aegis_resampled_harmful", action='store_true', help='use the smaller_aegis_resampled dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_word_balanced_benign", action='store_true', help='use the wildguard-word-balanced dataset for getting representations instead of the benign data')
    parser.add_argument("--use_wildguard_word_balanced_harmful", action='store_true', help='use the wildguard-word-balanced dataset for getting representations instead of the benign data')
    
    parser.add_argument("--use_oai_moderation_dataset_harmful", action='store_true', help='use the oai-moderation dataset for getting representations instead of the benign data')
    parser.add_argument("--use_harmbench", action='store_true', help='use the oai-moderation dataset for getting representations instead of the benign data')
    parser.add_argument("--use_forbidden_questions", action='store_true', help='use the oai-moderation dataset for getting representations instead of the benign data')
    parser.add_argument("--use_filtered_toxicchat_benign", action='store_true', help='use the filtered toxicchat dataset for getting representations instead of the benign data')
    parser.add_argument("--use_filtered_toxicchat_harmful", action='store_true', help='use the filtered toxicchat dataset for getting representations instead of the benign data')
    parser.add_argument("--use_simple_safety_tests_dataset", action='store_true', help='use the simple_safety_tests dataset for getting representations instead of the benign data')
    parser.add_argument("--use_saladbench_dataset", action='store_true', help='use the saladbench dataset for getting representations instead of the benign data')
    parser.add_argument("--use_aya_redteaming_dataset", action='store_true', help='use the ayaredteaming dataset for getting representations instead of the benign data')
    parser.add_argument("--use_thai_toxicity_tweets_benign", action='store_true', help='use the thai toxicity tweets dataset for getting representations instead of the benign data')
    parser.add_argument("--use_thai_toxicity_tweets_harmful", action='store_true', help='use the thai toxicity tweets dataset for getting representations instead of the benign data')
    parser.add_argument("--use_ukr_tweets_benign", action='store_true', help='use the ukr tweets dataset for getting representations instead of the benign data')
    parser.add_argument("--use_ukr_tweets_harmful", action='store_true', help='use the ukr tweets dataset for getting representations instead of the benign data')
    parser.add_argument("--use_advbench_dataset", action='store_true', help='use the advbench dataset for getting representations instead of the benign data')
    parser.add_argument("--use_toxicity_jigsaw_dataset_benign", action='store_true', help='use the toxic jigsaw dataset for getting representations instead of the benign data')
    parser.add_argument("--use_toxicity_jigsaw_dataset_harmful", action='store_true', help='use the toxic jigsaw dataset for getting representations instead of the benign data')
    parser.add_argument("--use_toxic_text_dataset_benign", action='store_true', help='use the toxic text dataset for getting representations instead of the benign data')
    parser.add_argument("--use_toxic_text_dataset_harmful", action='store_true', help='use the toxic text dataset for getting representations instead of the benign data')
    
    # 'code_attack_cipher', 'code_attack_python_list', 'code_attack_python_stack', 'code_attack_python_string'
    parser.add_argument('--use_code_attack_cipher', action='store_true', help='use the code attack cipher for the jailbreaking prompts')
    parser.add_argument('--use_code_attack_python_list', action='store_true', help='use the code attack python list for the jailbreaking prompts')
    parser.add_argument('--use_code_attack_python_stack', action='store_true', help='use the code attack python stack for the jailbreaking prompts')
    parser.add_argument('--use_code_attack_python_string', action='store_true', help='use the code attack python string for the jailbreaking prompts')
    parser.add_argument('--use_code_attack_python_list_benign', action='store_true', help='use the code attack python list for the jailbreaking prompts')
    parser.add_argument('--use_code_attack_python_stack_benign', action='store_true', help='use the code attack python stack for the jailbreaking prompts')
    parser.add_argument('--use_code_attack_python_string_benign', action='store_true', help='use the code attack python string for the jailbreaking prompts')
    parser.add_argument('--use_code_attack_python_list_benign_testset', action='store_true', help='use the code attack python list for the jailbreaking prompts')
    parser.add_argument('--use_code_attack_python_stack_benign_testset', action='store_true', help='use the code attack python stack for the jailbreaking prompts')
    parser.add_argument('--use_code_attack_python_string_benign_testset', action='store_true', help='use the code attack python string for the jailbreaking prompts')
    
    ## for Vision-Language datasets
    parser.add_argument("--use_mm_vet_dataset", action='store_true', help='use the mm-vet dataset for getting representations instead of the benign data')
    parser.add_argument("--use_mm_vet_v2_dataset", action='store_true', help='use the mm-vet dataset for getting representations instead of the benign data')
    parser.add_argument("--use_hades_dataset", action='store_true', help='use the hades dataset for getting representations instead of the hades (harmful) data')
    parser.add_argument("--use_mm_safetybench_dataset", action='store_true', help='use the mm-safetybench dataset for getting representations instead of the mm_safetybench (harmful) data')
    parser.add_argument("--mm_safetybench_category_index", type=int, default=1, help='choose the category index to use for the mm-safetybench dataset', choices=list(range(1, 14)))
    parser.add_argument("--mml_safebench_category_index", type=int, default=1, help='choose the category index to use for the mml-safebench dataset', choices=list(range(0, 10)))
    parser.add_argument("--use_vlsbench_dataset", action='store_true', help='use the vlsbench dataset for getting representations instead of the vlsbench (harmful) data')
    parser.add_argument("--use_mml_safebench_figstep_dataset", action='store_true', help='use the mml-safebench-figstep dataset for getting representations instead of the mml-safebench-figstep (harmful) data')
    
    parser.add_argument("--use_sst2_dataset", action='store_true', help='use the sst2 dataset for getting representations instead of the benign data')
    
    parser.add_argument('--run_on_cluster', action='store_true', help='run the code on the cluster')
    parser.add_argument('--model_layer', type=int, help='choose the model layer to use for training the classifier', default=None)
    parser.add_argument('--classifier_lang', type=str, help='choose the language to train the classifier on', default=None)
    parser.add_argument('--log_to_wandb', action='store_true', help='log the results to wandb')
    parser.add_argument('--specific_language', type=str, help='choose the specific language to use for the classifier', default=None)
    
    ## Settings for measuring the performance of the english classifier across different languages.
    parser.add_argument('--measure_classifier_multilingual_performance', action='store_true', help='measure the performance of the classifier across different languages')
    parser.add_argument('--plot_classifier_multilingual_performance', action='store_true', help='plot the performance of the classifier across different languages')
    
    parser.add_argument('--get_mean_classifier_performance', action='store_true', help='get the mean classifier performance across different languages')
    
    parser.add_argument('--get_bag_of_words_classifier', action='store_true', help='get the bag of words classifier')
    parser.add_argument('--jailbreak_dataset_bag_of_words', type=str, default=None)
    
    parser.add_argument('--convert_english_to_cipher_languages', action='store_true', help='convert the english dataset to the cipher languages')
    
    args = parser.parse_args()
    
    if not args.train_classifiers_on_representations:
        assert sum([args.use_benign_data, args.use_multijail_data, args.use_toxigen_data]) <= 1, "Please provide only one dataset to use. "
    
    assert sum([args.add_no_translator_command_english, args.add_no_translator_command_respective_language, args.add_internally_translate_but_output_same_language]) <= 1
    
    if args.produce_responses or args.judge_responses or args.parse_judgements or args.get_hidden_layer_representations or args.train_classifiers_on_representations:
        assert args.llm_model, "Please provide the model to use."
    # assert sum([args.produce_responses, args.judge_responses, args.test, args.parse_judgements, args.inspect_judgements, args.separate_query_and_responses, args.get_hidden_layer_representations]) == 1, "Please provide only one argument."
    
    if args.test:
        if args.llm_model == 'gpt-4-azure':
            gpt4_access = GPT4Access()
            print(gpt4_access.produce_response("What is the capital of Africa?"))
        elif args.llm_model == 'gpt-4o':
            gpt4o_access = GPT4OAccess()
            print(gpt4o_access.produce_response("Query: How murder victims keep their crimes done in secret Response: I'm sorry, but I can't fulfill this request."))
        elif args.llm_model == 'adversarial-gpt':
            gpt4_access = UnsafeGPT4Access()
            print(gpt4_access.produce_response("What is the capital of India?"))
        elif args.llm_model == 'llama3-8b':
            ckpt_dir = "llama3_git_repo/Meta-Llama-3-8B/"
            tokenizer_path = "llama3_git_repo/Meta-Llama-3-8B/tokenizer.model"
            llama3_access = Llama3Access(args, ckpt_dir, tokenizer_path)
            print(llama3_access.produce_response("What is the capital of India?"))
        elif args.llm_model == 'llama3-8b-instruct':
            ckpt_dir = "llama3_git_repo/Meta-Llama-3-8B-Instruct/"
            tokenizer_path = "llama3_git_repo/Meta-Llama-3-8B-Instruct/tokenizer.model"
            llama3_access = Llama3Access(args, ckpt_dir, tokenizer_path)
            dialog = [[{"role": "user", "content": "What is the capital of India?"}]]
            response = llama3_access.produce_chat_response(dialog)
            print(response)
        elif args.llm_model == 'llama3.1-8b-instruct':
            ckpt_dir = "llama3_git_repo/Meta-Llama-3.1-8B-Instruct/"
            tokenizer_path = "llama3_git_repo/Meta-Llama-3.1-8B-Instruct/tokenizer.model"
            llama3_access = Llama3Access(args, ckpt_dir, tokenizer_path)
            dialog = [[{"role": "user", "content": "What is the capital of Hungary?"}]]
            response = llama3_access.produce_chat_response(dialog)
            print(response)
        elif args.llm_model == 'llama3-70b':
            response = ollama.generate(model='llama3:70b-text', prompt='Why is the sky blue?', options={"num_predict": 20, "temperature": 0, "top_p": 0.1})
            print(response['response'])
        elif args.llm_model == 'llama3-70b-instruct':
            response = ollama.chat(model='llama3:70b', messages=[{'role': 'user', 'content': 'Who is the president of India?'}], options={"num_predict": 20, "temperature": 0, "top_p": 0.1})
            print(response['message']['content'])
        else:
            raise NotImplementedError
    elif args.produce_responses:
        produce_response_for_jailbreaking_prompts(args)
    elif args.judge_responses:
        judge_model_responses(args)
    elif args.parse_judgements:
        parse_judgements_and_assign_scores(args)
    elif args.inspect_judgements:
        inspect_judgements(args, lang='bn')
    elif args.separate_query_and_responses:
        separate_query_and_responses(args)
    elif args.get_hidden_layer_representations or args.offload_specific_layer_representations is not None:
        assert sum([args.use_benign_data, args.use_multijail_data, args.use_toxigen_data, args.use_random_sampled_toxigen_benign, args.use_random_sampled_toxigen_harmful, args.use_length_filtered_aegis_benign, args.use_length_filtered_aegis_harmful, args.use_jailbreakbench_dataset_benign, args.use_jailbreakbench_dataset_harmful, args.use_xstest_dataset_benign, args.use_xstest_dataset_harmful, args.use_beavertails_rlhf_dataset_benign, args.use_beavertails_rlhf_dataset_harmful, args.use_flores200_dataset, args.use_mm_vet_dataset, args.use_mm_vet_v2_dataset, args.use_hades_dataset, args.use_mm_safetybench_dataset, args.use_vlsbench_dataset, args.use_mml_safebench_figstep_dataset, args.use_polyguardmix_train_100K_benign, args.use_polyguardmix_train_100K_harmful, args.use_polyguardmix_train_500K_benign, args.use_polyguardmix_train_500K_harmful, args.use_polyguardmix_train_all_benign, args.use_polyguardmix_train_all_harmful, args.use_polyguardmix_test_benign_prompt, args.use_polyguardmix_test_harmful_prompt, args.use_polyguardmix_test_benign_response, args.use_polyguardmix_test_harmful_response,\
        args.use_wildguard_ar, args.use_wildguard_zh, args.use_wildguard_cs, args.use_wildguard_nl, args.use_wildguard_en, args.use_wildguard_fr, args.use_wildguard_de, args.use_wildguard_hi, args.use_wildguard_it, args.use_wildguard_ja, args.use_wildguard_ko, args.use_wildguard_po, args.use_wildguard_pt, args.use_wildguard_ru, args.use_wildguard_es, args.use_wildguard_sv, args.use_wildguard_th, args.use_wildguard_caesar, args.use_wildguard_caesar1, args.use_wildguard_leet, args.use_wildguard_vowel, args.use_wildguard_base64, args.use_wildguard_hexadecimal, args.use_wildguard_alphanumeric, args.use_wildguard_ascii, \
        args.use_wildguardmix_sampled_benign, args.use_wildguardmix_sampled_harmful, \
        args.use_wikidata_en, args.use_wikidata_fr, \
        args.use_filtered_multijail_dataset_en, args.use_filtered_multijail_dataset_ar, args.use_filtered_multijail_dataset_bn, args.use_filtered_multijail_dataset_it, args.use_filtered_multijail_dataset_jv, args.use_filtered_multijail_dataset_ko, args.use_filtered_multijail_dataset_sw, args.use_filtered_multijail_dataset_th, args.use_filtered_multijail_dataset_vi, args.use_filtered_multijail_dataset_zh, args.use_filtered_csrt_dataset, \
        args.use_xsafety_bn, args.use_xsafety_fr, args.use_xsafety_sp, args.use_xsafety_zh, args.use_xsafety_ar, args.use_xsafety_hi, args.use_xsafety_ja, args.use_xsafety_ru, args.use_xsafety_de, args.use_xsafety_en, args.use_rtplx_en, args.use_rtplx_others, \
        args.use_llm_lat_harmful, args.use_llm_lat_benign, \
        args.use_aegis2_LG_filtered_benign, args.use_aegis2_LG_filtered_harmful, \
        args.use_aegis2_random_sample_benign, args.use_aegis2_random_sample_harmful, \
        args.use_original_llm_lat_harmful, args.use_original_llm_lat_benign, \
        args.use_wildguard_word_balanced_benign, args.use_wildguard_word_balanced_harmful, \
        args.use_smaller_aegis_resampled_benign, args.use_smaller_aegis_resampled_harmful, \
        args.use_third_time_aegis_resampled_benign, args.use_third_time_aegis_resampled_harmful, \
        args.use_oai_moderation_dataset_harmful, args.use_harmbench, args.use_forbidden_questions, args.use_filtered_toxicchat_benign, args.use_filtered_toxicchat_harmful, args.use_simple_safety_tests_dataset, args.use_saladbench_dataset, args.use_aya_redteaming_dataset, args.use_thai_toxicity_tweets_benign, args.use_thai_toxicity_tweets_harmful, args.use_ukr_tweets_benign, args.use_ukr_tweets_harmful, args.use_advbench_dataset, args.use_toxicity_jigsaw_dataset_benign, args.use_toxicity_jigsaw_dataset_harmful, args.use_toxic_text_dataset_benign, args.use_toxic_text_dataset_harmful, \
        args.use_code_attack_cipher, args.use_code_attack_python_list, args.use_code_attack_python_stack, args.use_code_attack_python_string, args.use_code_attack_python_list_benign, args.use_code_attack_python_stack_benign, args.use_code_attack_python_string_benign, \
        args.use_code_attack_python_list_benign_testset, args.use_code_attack_python_stack_benign_testset, args.use_code_attack_python_string_benign_testset, \
        ]) == 1
        assert args.last_token_representations or args.avg_token_representations, "Please provide the last token or average token representations."
        get_hidden_layer_representations(args)
    elif args.get_benign_dataset:
        get_benign_dataset(args)
    elif args.train_classifiers_on_representations:
        ## here we will use both the benign and jailbreak data to train the classifiers.
        # args.model_layer is not None, "Please provide the model layer to use for training the classifier."
        args.classifier_lang is not None, "Please provide the language to train the classifier on."
        args.classifier_arch is not None, "Please provide the classifier type to use for training the classifier."
        train_classifiers_on_representations(args)
    elif args.measure_classifier_multilingual_performance:
        ## here will we use a classifier trained on one language to measure its performance on all other languages.
        args.model_layer is None, "We do not need the layer of the english classifier."
        args.classifier_lang == "en", "Please provide the language to train the classifier on."
        measure_classifier_multilingual_performance(args)
    elif args.plot_classifier_multilingual_performance:
        plot_classifier_multilingual_performance(args)
    elif args.get_mean_classifier_performance:
        get_mean_classifier_performance(args)
    elif args.get_bag_of_words_classifier:
        get_bag_of_words_classifier(args)
    elif args.convert_english_to_cipher_languages:
        assert sum([args.use_benign_data, args.use_multijail_data, args.use_toxigen_data, args.use_random_sampled_toxigen_benign, args.use_random_sampled_toxigen_harmful, args.use_length_filtered_aegis_benign, args.use_jailbreakbench_dataset_benign, args.use_xstest_dataset_benign, args.use_beavertails_rlhf_dataset_benign, args.use_sst2_dataset, args.use_flores200_dataset, args.use_polyguardmix_train_100K_benign, args.use_polyguardmix_train_100K_harmful, args.use_polyguardmix_train_500K_benign, args.use_polyguardmix_train_500K_harmful, args.use_polyguardmix_train_all_benign, args.use_polyguardmix_train_all_harmful, args.use_polyguardmix_test_benign_prompt, args.use_polyguardmix_test_harmful_prompt, args.use_polyguardmix_test_benign_response, args.use_polyguardmix_test_harmful_response, args.use_wildguard_ciphers, args.use_wikitext_sampled, args.use_llm_lat_harmful, args.use_llm_lat_benign, args.use_smaller_aegis_resampled_benign, args.use_smaller_aegis_resampled_harmful, args.use_harmbench, args.use_advbench_dataset, args.use_forbidden_questions, args.use_saladbench_dataset, args.use_simple_safety_tests_dataset, args.use_filtered_toxicchat_benign, args.use_toxicity_jigsaw_dataset_benign, args.use_toxic_text_dataset_benign]) == 1
        convert_english_to_cipher_languages(args)
    else:
        raise ValueError("Please provide the correct argument.")

