import os

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random, time
import pandas as pd
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


all_languages = ['en', 'fr', 'de', 'es', 'fa', 'ar', 'hr', 'ja', 'pl', 'ru', 'sv', 'th', 'hi', 'it', 'ko', 'bn', 'pt', 'zh', 'he', 'sr', 'da', 'tr', 'el', 'id', 'zu', 'hu', 'eu', 'sw', 'af', 'bs', 'lo', 'ro', 'sl', 'uk', 'fi', 'ms', 'jv', 'cy', 'bg', 'hy', 'is', 'vi', 'si', 'mi', 'gu', 'kn', 'mr', 'ta', 'am', 'te', 'no', 'cs', 'nl', 'caesar1', 'caesar2', 'caesar', 'caesar4', 'caesar5', 'caesar6', 'caesar7', 'caesarneg1', 'caesarneg2', 'caesarneg3', 'caesarneg4', 'caesarneg5', 'caesarneg6', 'caesarneg7', 'ascii', 'hexadecimal', 'base64', 'leet', 'vowel', 'alphanumeric']

cipher_languages = ['caesar1', 'caesar2', 'caesar', 'caesar4', 'caesar5', 'caesar6', 'caesar7', 'caesarneg1', 'caesarneg2', 'caesarneg3', 'caesarneg4', 'caesarneg5', 'caesarneg6', 'caesarneg7', 'ascii', 'hexadecimal', 'base64', 'leet', 'vowel', 'alphanumeric']

natural_languages = ['en', 'fr', 'de', 'es', 'fa', 'ar', 'hr', 'ja', 'pl', 'ru', 'sv', 'th', 'hi', 'it', 'ko', 'bn', 'pt', 'zh', 'he', 'sr', 'da', 'tr', 'el', 'id', 'zu', 'hu', 'eu', 'sw', 'af', 'bs', 'lo', 'ro', 'sl', 'uk', 'fi', 'ms', 'jv', 'cy', 'bg', 'hy', 'is', 'vi', 'si', 'mi', 'gu', 'kn', 'mr', 'ta', 'am', 'te', 'no', 'cs', 'nl']
languages_names = {'en': 'English', 'fr': 'French', 'de': 'German', 'es': 'Spanish', 'fa': 'Farsi', 'ar': 'Arabic', 'hr': 'Croatian', 'ja': 'Japanese', 'pl': 'Polish', 'ru': 'Russian', 'sv': 'Swedish', 'th': 'Thai', 'hi': 'Hindi', 'bn': 'Bengali', 'it': 'Italian', 'ko': 'Korean', 'pt': 'Portuguese', 'te': 'Telugu', 'vi': 'Vietnamese', 'zh': 'Chinese', 'he': 'Hebrew', 'sr': 'Serbian', 'da': 'Danish', 'tr': 'Turkish', 'el': 'Greek', 'id': 'Indonesian', 'nl': 'Dutch',
                   'ascii': 'ascii encoding (characters converted to ascii values)',
                   'hexadecimal': 'hexadecimal encoding (characters converted to hexadecimal values)', 
                   'base64': 'base64 encoding', 
                   'leet': 'leet encoding (characters replaced by leet characters)', 
                   'vowel': 'vowel encoding (vowels removed)',
                   'alphanumeric': 'alphanumeric encoding (english letters converted to ascii values)', 
                   'caesar2': 'caesar2 encoding (english characters shifted by +2)',
                   'caesar': 'caesar encoding (english characters shifted by +3)', 
                   'caesar1': 'caesar1 encoding (english characters shifted by +1)',
                   'caesar4': 'caesar4 encoding (english characters shifted by +4)', 
                   'caesar5': 'caesar5 encoding (english characters shifted by +5)',
                   'caesar6': 'caesar6 encoding (english characters shifted by +6)',
                   'caesar7': 'caesar7 encoding (english characters shifted by +7)',
                   'caesar8': 'caesar8 encoding (english characters shifted by +8)',
                   'caesar9': 'caesar9 encoding (english characters shifted by +9)',
                   'caesarneg1': 'caesarneg1 encoding (english characters shifted by -1)',
                   'caesarneg2': 'caesarneg2 encoding (english characters shifted by -2)',
                   'caesarneg3': 'caesarneg3 encoding (english characters shifted by -3)',
                   'caesarneg4': 'caesarneg4 encoding (english characters shifted by -4)',
                   'caesarneg5': 'caesarneg5 encoding (english characters shifted by -5)',
                   'caesarneg6': 'caesarneg6 encoding (english characters shifted by -6)',
                   'caesarneg7': 'caesarneg7 encoding (english characters shifted by -7)',
                   'caesarneg8': 'caesarneg8 encoding (english characters shifted by -8)',
                   'caesarneg9': 'caesarneg9 encoding (english characters shifted by -9)',
                   'unicode': 'Unicode', 
                   'utf': 'Utf', 
                   'atbash': 'atbash encoding (english characters mirrored, so a becomes z, b becomes y, c becomes x, etc.)',
                   'morse': 'morse encoding (english characters converted to morse code)', 
                   'vigenere': 'vigenere encoding (english characters shifted by some key)',
                   'keyboard': 'keyboard encoding (shift right for QWERTY layout)', 
                   'reverse': 'reverse encoding (english characters reversed)'
                   }

three_letter_iso_code = ['eng', 'fra', 'deu', 'spa', 'fas', 'ara', 'hrv', 'jpn', 'pol', 'rus', 'swe', 'tha', 'hin', 'ben', 'ita', 'kor', 'por', 'tel', 'vie', 'zho', 'heb', 'srp', 'dan', 'tur', 'ell', 'ind']

assert len(all_languages) == 73 == len(cipher_languages) + len(natural_languages)


def get_split_type(args):
    if args.split_train_test_along_only_languages:
        split_type = 'only_languages'
    elif args.split_train_test_along_languages_and_datapoints:
        split_type = 'languages_and_datapoints'
    elif args.split_train_test_along_languages_and_datapoints_20_train:
        split_type = 'languages_and_datapoints_20_train'
    elif args.split_train_test_along_languages_and_datapoints_30_train:
        split_type = 'languages_and_datapoints_30_train'
    elif args.split_train_test_along_languages_and_datapoints_40_train:
        split_type = 'languages_and_datapoints_40_train'
    elif args.split_train_test_along_languages_and_datapoints_50_train:
        split_type = 'languages_and_datapoints_50_train'
    elif args.split_train_test_along_languages_and_datapoints_60_train:
        split_type = 'languages_and_datapoints_60_train'
    elif args.split_train_test_along_languages_and_datapoints_10_train:
        split_type = 'languages_and_datapoints_10_train'
    elif args.split_train_test_along_languages_and_datapoints_5_train:
        split_type = 'languages_and_datapoints_5_train'
    elif args.split_train_test_along_languages_and_datapoints_2_train:
        split_type = 'languages_and_datapoints_2_train'
    elif args.split_train_test_along_languages_and_datapoints_1_train:
        split_type = 'languages_and_datapoints_1_train'
    elif args.split_train_test_along_languages_and_datapoints_50_datapoints:
        split_type = 'languages_and_datapoints_50_datapoints'
    elif args.split_train_test_along_languages_and_datapoints_100_datapoints:
        split_type = 'languages_and_datapoints_100_datapoints'
    else:
        raise NotImplementedError
    return split_type


import hashlib

def stable_hash(value):
    return int(hashlib.sha256(value.encode('utf-8')).hexdigest(), 16)


def deterministic_shuffle(items, seed):
    items = items[:]
    for i in range(len(items)-1, 0, -1):
        # Use stable_hash to get a consistent hash value
        j = stable_hash(f'{seed}-{i}') % (i+1)
        items[i], items[j] = items[j], items[i]
    return items


def init_wandb_and_log(args, project_name=None, run_name=None):
    os.environ["WANDB_API_KEY"] = "65b10491413acd011c96d46acd3990854fded569"
    import wandb as wandb_logging
    import copy
    
    if project_name is None:
        if args.split_train_test_along_only_languages:
            project_name = f'accuracy-after-splitting-along-language{"-logistic" if args.train_classifier_harmfulness_logistic else "mlp"}-{args.llm_model}'
        elif args.split_train_test_along_languages_and_datapoints_50_datapoints or args.split_train_test_along_languages_and_datapoints_100_datapoints:
            project_name = f'accuracy-after-splitting-along-language-and-datapoints-50-datapoints{"-logistic" if args.train_classifier_harmfulness_logistic else "-mlp"}-{args.llm_model}'
        else:
            project_name = f'check-accuracy-after-splitting-along-language-and-datapoints{"-logistic" if args.train_classifier_harmfulness_logistic else "mlp"}-{args.llm_model}'

    split_type = get_split_type(args)
    
    if run_name is None:
        if args.compute_CS_LRD_embeddings_classifier:
            run_name = f"mlp_classifier_{args.embeddings_dataset}_split_type_{split_type}-seed_{args.random_seed_for_language_split}_{args.random_seed_for_datapoint_split}_CS_LRD"
        else:       ## This is the for the vanilla classifier
            run_name = f"{'logistic' if args.train_classifier_harmfulness_logistic else 'mlp'}_classifier_{args.embeddings_dataset}_split_type_{split_type}-seed_{args.random_seed}" if args.train_classifier_harmfulness_logistic else f"vanilla_classifier_{args.embeddings_dataset}_split_type_{split_type}-seed_{args.random_seed_for_language_split}_{args.random_seed_for_datapoint_split}"

    wandb_logging.init(project=project_name,
        name = run_name,
        config={
                "num_epochs": args.num_epochs if not args.train_classifier_harmfulness_logistic else 5000,
                "start_learning_rate": copy.deepcopy(args.learning_rate),
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "embeddings_dataset": args.embeddings_dataset,
                "random_seed_language_split": args.random_seed_for_language_split,
                "random_seed_datapoint_split": args.random_seed_for_datapoint_split,
                "split_type": split_type,
                "hidden_states_representations_layer": args.hidden_states_representations_layer,
                "llm_model": args.llm_model,
            }
        )

    return wandb_logging


def load_embeddings(args, split_along_languages, split_along_datapoints, split_along_languages_and_datapoints, load_separately=False):

    global all_languages, cipher_languages, natural_languages
    assert sum([split_along_languages, split_along_datapoints, split_along_languages_and_datapoints, load_separately]) == 1
    assert args.hidden_states_representations_layer is not None
    
    if split_along_datapoints:
        if args.custom_train_languages is not None:     ## since we are splitting only along datapoints, we must be having only train languages
            train_languages = args.custom_train_languages
            all_languages = train_languages
            assert len(all_languages) > 0
            print(f"Train languages: {train_languages}")

    elif args.split_languages_randomly:   ## choose half of the langauges to be the training set, and the other half to be the test set
        assert args.random_seed_for_language_split is not None
        cipher_languages = deterministic_shuffle(cipher_languages, seed=args.random_seed_for_language_split)
        natural_languages = deterministic_shuffle(natural_languages, seed=args.random_seed_for_language_split)
        # Now, we have 26 natural and 24 cipher languages in total. We will use 13 natural + 12 cipher for training. 6 natural + 4 cipher for validation. 7 natural + 8 cipher for testing. We will randomly shuffle the languages and then split them into train, validation and test sets.
        train_languages = natural_languages[:13] + cipher_languages[:12]
        validation_languages = natural_languages[13:19] + cipher_languages[12:16]
        test_languages = natural_languages[19:] + cipher_languages[16:]
        assert len(train_languages) == 25 and len(validation_languages) == 10 and len(test_languages) == 15
        print(f"random seed: {args.random_seed_for_language_split}, Train languages: {train_languages}, Test languages: {test_languages}")
    
    elif args.split_languages_randomly_test:   ## choose half of the langauges to be the training set, and the other half to be the test set
        assert args.random_seed_for_language_split is not None
        cipher_languages = deterministic_shuffle(cipher_languages, seed=args.random_seed_for_language_split)
        natural_languages = deterministic_shuffle(natural_languages, seed=args.random_seed_for_language_split)
        # Now, we have 26 natural and 24 cipher languages in total. We will use 13 natural + 12 cipher for training. 6 natural + 4 cipher for validation. 7 natural + 8 cipher for testing. We will randomly shuffle the languages and then split them into train, validation and test sets.
        train_languages = natural_languages[:4]
        validation_languages = natural_languages[4:6]
        test_languages = natural_languages[6:8]
        all_languages = train_languages + validation_languages + test_languages
    
    elif args.split_only_cipher_languages_randomly:   ## choose half of the cipher languages to be the training set, and the other half of the cipher to be the test set, no natural languages
        assert args.random_seed_for_language_split is not None
        cipher_languages = deterministic_shuffle(cipher_languages, seed=args.random_seed_for_language_split)
        num_cipher_languages = len(cipher_languages)
        num_train_cipher_languages = int(num_cipher_languages * 0.5)
        num_validation_cipher_languages = int(num_cipher_languages * 0.25)
        num_test_cipher_languages = num_cipher_languages - num_train_cipher_languages - num_validation_cipher_languages
        train_languages = cipher_languages[:num_train_cipher_languages]
        validation_languages = cipher_languages[num_train_cipher_languages:num_train_cipher_languages + num_validation_cipher_languages]
        test_languages = cipher_languages[num_train_cipher_languages + num_validation_cipher_languages:]
        all_languages = train_languages + validation_languages + test_languages
        assert len(test_languages) == num_test_cipher_languages
        print(f"random seed: {args.random_seed_for_language_split}, Train languages: {train_languages}, Test languages: {test_languages}")

    elif args.split_custom_languages_for_train_test:
        train_languages = args.custom_train_languages        ## this is a list of languages
        test_languages = args.custom_test_languages          ## this is a list of languages
        if args.custom_validation_languages is None:
            validation_languages = ['en']
        else:
            validation_languages = args.custom_validation_languages
        print(f"Train languages: {train_languages}, Validation languages: {validation_languages}, Test languages: {test_languages}")
        all_languages = list(set(train_languages + validation_languages + test_languages))

    elif args.embeddings_dataset in ["hades_dataset", "mm_safetybench_dataset", "vlsbench_dataset", "mml_safebench_figstep_dataset", "polyguardmix_train_100K", "polyguardmix_train_500K", "polyguardmix_train_all", "polyguardmix_train_10K"]:
        pass        ## here there is only english
    
    else:
        raise NotImplementedError
    
    if args.embeddings_dataset == "toxigen_dataset":
        model_representations_directory = "randomly_sampled_toxigen_dataset/model_representations_random_sampled_toxigen"
        if args.use_last_token_embeddings:
            model_representations_directory = "randomly_sampled_toxigen_dataset/model_last_token_representations_random_sampled_toxigen"
        directory_name = "randomly_sampled_toxigen"
        badname = "toxic"
        num_datapoints_total = 2000
    elif args.embeddings_dataset == "jailbreakbench_dataset":
        model_representations_directory = "jailbreakbench_dataset/model_representations_multilingual_jailbreaks"
        directory_name = "jailbreakbench_dataset"
        badname = "harmful"
        num_datapoints_total = 200
    elif args.embeddings_dataset == "xstest_dataset":
        model_representations_directory = "xstest_dataset/model_representations_multilingual_jailbreaks"
        directory_name = "xstest_dataset"
        badname = "harmful"
        num_datapoints_total = 400
    elif args.embeddings_dataset == "aegis_safety_dataset":
        model_representations_directory = "aegis_safety_dataset/model_representations_multilingual_jailbreaks"
        directory_name = "aegis_safety_dataset"
        badname = "harmful"
        num_datapoints_total = 2536
    elif args.embeddings_dataset == "beavertails_rlhf_dataset":
        model_representations_directory = "beavertails_rlhf_dataset/model_representations_multilingual_jailbreaks"
        directory_name = "beavertails_rlhf_dataset"
        badname = "harmful"
        num_datapoints_total = 4000
    elif args.embeddings_dataset == "hades_dataset":
        model_representations_directory = "mm_vet_dataset/model_representations_hades_dataset"
        directory_name = "hades_dataset"
        all_languages = ["en"]
        num_datapoints_total = 750 * 2
    elif args.embeddings_dataset == "mm_safetybench_dataset":
        model_representations_directory = "mm_vet_dataset/model_representations_mm_safetybench_dataset"
        directory_name = "mm_safetybench_dataset"
        all_languages = ["en"]
        num_datapoints_total = 1680 * 2
    elif args.embeddings_dataset == "vlsbench_dataset":
        model_representations_directory = "vlsbench_dataset/model_representations_multilingual_jailbreaks"
        directory_name = "vlsbench_dataset"
        all_languages = ["en"]
        num_datapoints_total = 2241 * 2
    elif args.embeddings_dataset == "mml_safebench_figstep_dataset":
        model_representations_directory = "mml_safebench_figstep_dataset/model_representations_mml_safebench_figstep_dataset"
        directory_name = "mml_safebench_figstep_dataset"
        all_languages = ["en"]
        num_datapoints_total = 517 * 2
    elif args.embeddings_dataset == "polyguardmix_train_100K" or args.embeddings_dataset == "polyguardmix_train_500K" or args.embeddings_dataset == "polyguardmix_train_all" or args.embeddings_dataset == "polyguardmix_train_10K":
        model_representation_directory = "polyguardmix_dataset/model_representations_multilingual_jailbreaks"
        directory_name = "polyguardmix_dataset"
        all_languages = ["en"]
        assert args.load_separate_test_set_name is not None
        if args.embeddings_dataset == "polyguardmix_train_10K":
            num_datapoints_total = 200
        elif args.embeddings_dataset == "polyguardmix_train_100K":
            num_datapoints_total = 100000
        elif args.embeddings_dataset == "polyguardmix_train_500K":
            num_datapoints_total = 500000
        elif args.embeddings_dataset == "polyguardmix_train_all":
            num_datapoints_total = 2988961
    elif args.embeddings_dataset == "llm_lat_dataset":
        model_representations_directory = "llm_lat_dataset/model_representations_multilingual_jailbreaks/layer_wise_representations"
        directory_name = "llm_lat_dataset"
        badname = "harmful"
        num_datapoints_total = 21981
    elif args.embeddings_dataset == "aegis2_safety_llamaguard_filtered":
        model_representations_directory = "Aegis-AI-Content-Safety-Dataset-2.0/llamaguard_filtered_sample/model_representations_multilingual_jailbreaks"
        directory_name = "aegis2_dataset"
        badname = "harmful"
        num_datapoints_total = 9994
    elif args.embeddings_dataset == "aegis2_safety_random_sampled":
        model_representations_directory = "Aegis-AI-Content-Safety-Dataset-2.0/random_5K_sample/model_representations_multilingual_jailbreaks"
        directory_name = "aegis2_dataset"
        badname = "harmful"
        num_datapoints_total = 9997
    elif args.embeddings_dataset == "wildguard_word_balanced_random_sampled":
        model_representations_directory = "wildguard_datasets/word_balanced_sample_5K/model_representations_multilingual_jailbreaks"
        directory_name = "wildguard_balanced"
        badname = "harmful"
        num_datapoints_total = 11193
    elif args.embeddings_dataset == "original_smaller_llm_lat_dataset":
        model_representations_directory = "llm_lat_dataset/original_llm_lat_dataset/model_representations_multilingual_jailbreaks"
        directory_name = "original_llm_lat_dataset"
        badname = "harmful"
        num_datapoints_total = 9894
    elif args.embeddings_dataset == "second_time_aegis_smaller_sampled":
        model_representations_directory = "second_time_aegis_safety_dataset/model_representations_multilingual_jailbreaks"
        directory_name = "smaller_aegis_resampled"
        badname = "harmful"
        num_datapoints_total = 2858
    elif args.embeddings_dataset == "third_time_aegis2_resampled":
        model_representations_directory = "Aegis-AI-Content-Safety-Dataset-2.0/third_time_aegis2_sampling/model_representations_multilingual_jailbreaks"
        directory_name = "third_time_aegis_resampled"
        badname = "harmful"
        num_datapoints_total = 9970
    else:
        raise NotImplementedError
    
    train_embeddings = {}
    validation_embeddings = {}
    test_embeddings = {}
    add_to_all_languages = []
    
    if split_along_datapoints or split_along_languages_and_datapoints:
        ## get the order of deterministic shuffle and use the same for all languages
        shuffled_order_datapoints = deterministic_shuffle(list(range(num_datapoints_total)), seed=args.random_seed_for_datapoint_split)
    
    if args.split_datapoints_for_english_only:
        all_languages = args.other_language_translate_to_english + ['en']
    if args.split_datapoints_for_these_languages_only:
        all_languages = args.other_language_translate_to_english
        assert len(all_languages) > 0

    for language in all_languages:
        if args.embeddings_dataset == "hades_dataset" or args.embeddings_dataset == "mm_safetybench_dataset" or args.embeddings_dataset == "vlsbench_dataset" or args.embeddings_dataset == "mml_safebench_figstep_dataset":
            if args.embeddings_dataset == "hades_dataset" or args.embeddings_dataset == "vlsbench_dataset":
                model_embeddings_toxic = torch.load(f"mm_vet_dataset/model_representations_{args.embeddings_dataset}/{args.embeddings_dataset}_benign_representations_{args.llm_model}_{language}.pt", weights_only=True)
            elif args.embeddings_dataset == "mm_safetybench_dataset" or args.embeddings_dataset == "mml_safebench_figstep_dataset":
                if args.embeddings_dataset == "mm_safetybench_dataset":
                    range_here = range(1, 14)
                elif args.embeddings_dataset == "mml_safebench_figstep_dataset":
                    range_here = range(0, 10)
                all_model_embeddings_toxic = []
                for category_index in range_here:
                    model_embeddings_toxic_index = torch.load(f"mm_vet_dataset/model_representations_{args.embeddings_dataset}/{args.embeddings_dataset}_{category_index}_benign_representations_{args.llm_model}_{language}.pt", weights_only=True)
                    all_model_embeddings_toxic.append(model_embeddings_toxic_index)
                model_embeddings_toxic = torch.cat(all_model_embeddings_toxic, dim=0)
            else:
                raise NotImplementedError
            # model_embeddings_benign = torch.load(f"mm_vet_dataset/model_representations_mm_vet_dataset/mm_vet_dataset_benign_representations_{args.llm_model}_{language}.pt", weights_only=True)
            model_embeddings_benign = torch.load(f"mm_vet_dataset/model_representations_mm_vet_v2_dataset/mm_vet_v2_dataset_benign_representations_{args.llm_model}_{language}.pt", weights_only=True)
            ## make sure that the number of embeddings is the same for both toxic and benign, upsample the smaller one to ensure that
            if model_embeddings_benign.shape[0] < model_embeddings_toxic.shape[0]:
                ## upsample the benign embeddings
                model_embeddings_benign = torch.cat([model_embeddings_benign, model_embeddings_benign[torch.randint(0, model_embeddings_benign.shape[0], (model_embeddings_toxic.shape[0] - model_embeddings_benign.shape[0],))]], dim=0)
            elif model_embeddings_toxic.shape[0] < model_embeddings_benign.shape[0]:
                ## upsample the toxic embeddings
                model_embeddings_toxic = torch.cat([model_embeddings_toxic, model_embeddings_toxic[torch.randint(0, model_embeddings_toxic.shape[0], (model_embeddings_benign.shape[0] - model_embeddings_toxic.shape[0],))]], dim=0)
        
        elif args.embeddings_dataset == "polyguardmix_train_10K" or args.embeddings_dataset == "polyguardmix_train_100K" or args.embeddings_dataset == "polyguardmix_train_500K" or args.embeddings_dataset == "polyguardmix_train_all":
        
            if args.embeddings_dataset == "polyguardmix_train_10K" or args.embeddings_dataset == "polyguardmix_train_100K" or args.embeddings_dataset == "polyguardmix_train_500K":
                # if args.embeddings_dataset == "polyguardmix_train_10K":
                #     model_embeddings_toxic = torch.load(f"{model_representation_directory}/polyguardmix_train_100K_harmful_{language}_representations_{args.llm_model}_{language}.pt", weights_only=True)
                #     print(f"loaded {model_embeddings_toxic.shape} toxic embeddings")
                #     model_embeddings_toxic = model_embeddings_toxic[:100, :].clone()
                #     print(f"saving {model_embeddings_toxic.shape} toxic embeddings")
                #     torch.save(model_embeddings_toxic, f"{model_representation_directory}/polyguardmix_train_10K_harmful_{language}_representations_{args.llm_model}_{language}.pt")
                #     model_embeddings_benign = torch.load(f"{model_representation_directory}/polyguardmix_train_100K_benign_{language}_representations_{args.llm_model}_{language}.pt", weights_only=True)
                #     print(f"loaded {model_embeddings_benign.shape} benign embeddings")
                #     model_embeddings_benign = model_embeddings_benign[:100, :].clone()
                #     print(f"saving {model_embeddings_benign.shape} benign embeddings")
                #     torch.save(model_embeddings_benign, f"{model_representation_directory}/polyguardmix_train_10K_benign_{language}_representations_{args.llm_model}_{language}.pt")
                #     print("saved the 10K embeddings")
                #     exit()
                # else:
                model_embeddings_toxic = torch.load(f"{model_representation_directory}/{args.embeddings_dataset}_harmful_{language}_representations_{args.llm_model}_{language}.pt", weights_only=True)
                model_embeddings_benign = torch.load(f"{model_representation_directory}/{args.embeddings_dataset}_benign_{language}_representations_{args.llm_model}_{language}.pt", weights_only=True)
                
                model_embeddings_toxic = model_embeddings_toxic[:, args.hidden_states_representations_layer, :].clone()
                model_embeddings_benign = model_embeddings_benign[:, args.hidden_states_representations_layer, :].clone()

                merged_embeddings = torch.cat([model_embeddings_toxic, model_embeddings_benign], dim=0)
                merged_labels = torch.cat([torch.ones(model_embeddings_toxic.shape[0], 1, dtype=torch.int), torch.zeros(model_embeddings_benign.shape[0], 1, dtype=torch.int)], dim=0)
                assert merged_embeddings.shape[0] == merged_labels.shape[0] == num_datapoints_total, f"merged_embeddings.shape: {merged_embeddings.shape}, merged_labels.shape: {merged_labels.shape}, num_datapoints_total: {num_datapoints_total}"
                print(f"loaded train embeddings using hidden layer {args.hidden_states_representations_layer}")
                
                train_embeddings[language] = {}
                train_embeddings[language]['toxic'] = model_embeddings_toxic
                train_embeddings[language]['benign'] = model_embeddings_benign
                
            elif args.embeddings_dataset == "polyguardmix_train_all":
                ## here we divived the dataset in 10 chunks for both benign and toxic, we will load all the chunks and then concatenate them
                overall_model_embeddings_benign, overall_model_embeddings_toxic = [], []
                for chunk_idx in range(10):
                    model_embeddings_toxic = torch.load(f"{model_representation_directory}/{args.embeddings_dataset}_harmful_chunk_{chunk_idx}_{language}_representations_{args.llm_model}_{language}.pt", weights_only=True)
                    model_embeddings_benign = torch.load(f"{model_representation_directory}/{args.embeddings_dataset}_benign_chunk_{chunk_idx}_{language}_representations_{args.llm_model}_{language}.pt", weights_only=True)
                    model_embeddings_toxic = model_embeddings_toxic[:, args.hidden_states_representations_layer, :].clone()
                    model_embeddings_benign = model_embeddings_benign[:, args.hidden_states_representations_layer, :].clone()
                    overall_model_embeddings_toxic.append(model_embeddings_toxic)
                    overall_model_embeddings_benign.append(model_embeddings_benign)
                    print(f"loaded train embeddings using hidden layer {args.hidden_states_representations_layer} for chunk {chunk_idx}")
                model_embeddings_toxic = torch.cat(overall_model_embeddings_toxic, dim=0)
                model_embeddings_benign = torch.cat(overall_model_embeddings_benign, dim=0)
                merged_embeddings = torch.cat([model_embeddings_toxic, model_embeddings_benign], dim=0)
                merged_labels = torch.cat([torch.ones(model_embeddings_toxic.shape[0], 1, dtype=torch.int), torch.zeros(model_embeddings_benign.shape[0], 1, dtype=torch.int)], dim=0)
                assert merged_embeddings.shape[0] == merged_labels.shape[0] == num_datapoints_total, f"merged_embeddings.shape: {merged_embeddings.shape}, merged_labels.shape: {merged_labels.shape}, num_datapoints_total: {num_datapoints_total}"
                print(f"loaded train embeddings using hidden layer {args.hidden_states_representations_layer}")
                train_embeddings[language] = {}
                train_embeddings[language]['toxic'] = model_embeddings_toxic
                train_embeddings[language]['benign'] = model_embeddings_benign
            
            if args.load_separate_test_set_name == 'polyguardmix_test_prompts' or args.load_separate_test_set_name == 'polyguardmix_test_responses':
                ## now load the test embeddings
                if args.load_separate_test_set_name == 'polyguardmix_test_prompts':
                    model_embeddings_toxic_test = torch.load(f"{model_representation_directory}/polyguardmix_test_harmful_prompt_{language}_representations_{args.llm_model}_{language}.pt", weights_only=True)
                    model_embeddings_benign_test = torch.load(f"{model_representation_directory}/polyguardmix_test_benign_prompt_{language}_representations_{args.llm_model}_{language}.pt", weights_only=True)
                elif args.load_separate_test_set_name == 'polyguardmix_test_responses':
                    model_embeddings_toxic_test = torch.load(f"{model_representation_directory}/polyguardmix_test_harmful_response_{language}_representations_{args.llm_model}_{language}.pt", weights_only=True)
                    model_embeddings_benign_test = torch.load(f"{model_representation_directory}/polyguardmix_test_benign_response_{language}_representations_{args.llm_model}_{language}.pt", weights_only=True)
                
                model_embeddings_toxic_test = model_embeddings_toxic_test[:, args.hidden_states_representations_layer, :].clone()
                model_embeddings_benign_test = model_embeddings_benign_test[:, args.hidden_states_representations_layer, :].clone()
                
                merged_embeddings_test = torch.cat([model_embeddings_toxic_test, model_embeddings_benign_test], dim=0)
                merged_labels_test = torch.cat([torch.ones(model_embeddings_toxic_test.shape[0], 1, dtype=torch.int), torch.zeros(model_embeddings_benign_test.shape[0], 1, dtype=torch.int)], dim=0)
                assert merged_embeddings_test.shape[0] == merged_labels_test.shape[0]
                test_embeddings[language] = {}
                test_embeddings[language]['toxic'] = model_embeddings_toxic_test
                test_embeddings[language]['benign'] = model_embeddings_benign_test
                print(f"loaded polyguardmix test embeddings for {args.load_separate_test_set_name} using hidden layer {args.hidden_states_representations_layer}")
            
            elif args.load_separate_test_set_name == 'multijail_dataset':
                for test_lang in ['en', 'bn', 'ar', 'it', 'jv', 'ko', 'sw', 'th', 'vi', 'zh']:
                    model_embeddings_toxic_test = torch.load(f"multijail_dataset/model_representations_multilingual_jailbreaks/multijail_dataset_{test_lang}_representations_{args.llm_model}_{test_lang}.pt", weights_only=True)
                    model_embeddings_toxic_test = model_embeddings_toxic_test[:, args.hidden_states_representations_layer, :].clone()
                    # merged_embeddings_test = model_embeddings_toxic_test
                    # merged_labels_test = torch.ones(model_embeddings_toxic_test.shape[0], 1, dtype=torch.int)
                    ## use first 50% for validation and remaining for test
                    # val_index = int(model_embeddings_toxic_test.shape[0] * 0.5)
                    # validation_embeddings[test_lang] = {}
                    # validation_embeddings[test_lang]['toxic'] = model_embeddings_toxic_test[val_index:, :]
                    test_embeddings[test_lang] = {}
                    # test_embeddings[test_lang]['toxic'] = model_embeddings_toxic_test[:val_index, :]
                    test_embeddings[test_lang]['toxic'] = model_embeddings_toxic_test[:, :]
                    # test_embeddings[test_lang]['benign'] = model_embeddings_toxic_test
                    print(f"loaded multijail test embeddings for {test_lang}")
                    ## in this dataset all the embeddings are harmful. Also we need to update the all_languages. 
                    if test_lang != 'en':
                        add_to_all_languages.append(test_lang)
                
            elif args.load_separate_test_set_name == 'csrt_dataset':
                model_embeddings_toxic_test = torch.load(f"CSRT_dataset/model_representations_multilingual_jailbreaks/csrt_dataset_representations_{args.llm_model}_code_switched.pt", weights_only=True)
                model_embeddings_toxic_test = model_embeddings_toxic_test[:, args.hidden_states_representations_layer, :].clone()
                # merged_embeddings_test = model_embeddings_toxic_test
                # merged_labels_test = torch.ones(model_embeddings_toxic_test.shape[0], 1, dtype=torch.int)
                ## use first 50% for validation and remaining for test
                val_index = int(model_embeddings_toxic_test.shape[0] * 0.5)
                validation_embeddings[language] = {}
                validation_embeddings[language]['toxic'] = model_embeddings_toxic_test[:val_index, :]
                test_embeddings[language] = {}
                test_embeddings[language]['toxic'] = model_embeddings_toxic_test[val_index:, :]
                # test_embeddings[language]['benign'] = model_embeddings_toxic_test
                print("loaded csrt test embeddings")
                ## in this dataset all the embeddings are harmful. Since this is just one language, that is code_switched, we will not add it to the all_languages.
            
            elif args.load_separate_test_set_name == 'xsafety_dataset':
                for test_lang in ['ar', 'bn', 'de', 'en', 'fr', 'hi', 'ja', 'ru', 'sp', 'zh']:
                    model_embeddings_toxic_test = torch.load(f"xsafety_dataset/model_representations_multilingual_jailbreaks/xsafety_dataset_representations_{args.llm_model}_{test_lang}.pt", weights_only=True)
                    model_embeddings_toxic_test = model_embeddings_toxic_test[:, args.hidden_states_representations_layer, :].clone()
                    # merged_embeddings_test = model_embeddings_toxic_test
                    # merged_labels_test = torch.ones(model_embeddings_toxic_test.shape[0], 1, dtype=torch.int)
                    ## use first 50% for validation and remaining for test
                    val_index = int(model_embeddings_toxic_test.shape[0] * 0.5)
                    validation_embeddings[test_lang] = {}
                    validation_embeddings[test_lang]['toxic'] = model_embeddings_toxic_test[val_index:, :]
                    test_embeddings[test_lang] = {}
                    test_embeddings[test_lang]['toxic'] = model_embeddings_toxic_test[:val_index, :]
                    # test_embeddings[test_lang]['benign'] = model_embeddings_toxic_test
                    print(f"loaded multijail test embeddings for {test_lang}")
                    ## in this dataset all the embeddings are harmful. Also we need to update the all_languages. 
                    if test_lang != 'en':
                        add_to_all_languages.append(test_lang)
                        
            elif args.load_separate_test_set_name == 'rtplx':
                for test_lang in ['en', 'others']:
                    model_embeddings_toxic_test = torch.load(f"rtp_lx_dataset/model_representations_multilingual_jailbreaks/rtp_lx_dataset_representations_{args.llm_model}_{test_lang}.pt", weights_only=True)
                    model_embeddings_toxic_test = model_embeddings_toxic_test[:, args.hidden_states_representations_layer, :].clone()
                    # merged_embeddings_test = model_embeddings_toxic_test
                    # merged_labels_test = torch.ones(model_embeddings_toxic_test.shape[0], 1, dtype=torch.int)
                    ## use first 50% for validation and remaining for test
                    val_index = int(model_embeddings_toxic_test.shape[0] * 0.5)
                    validation_embeddings[test_lang] = {}
                    validation_embeddings[test_lang]['toxic'] = model_embeddings_toxic_test[val_index:, :]
                    test_embeddings[test_lang] = {}
                    test_embeddings[test_lang]['toxic'] = model_embeddings_toxic_test[:val_index, :]
                    # test_embeddings[test_lang]['benign'] = model_embeddings_toxic_test
                    print(f"loaded multijail test embeddings for {test_lang}")
                    ## in this dataset all the embeddings are harmful. Also we need to update the all_languages. 
                    if test_lang != 'en':
                        add_to_all_languages.append(test_lang)

            else:
                raise NotImplementedError

            if language not in validation_embeddings:
                validation_embeddings[language] = {}

        elif args.embeddings_dataset == "beavertails_rlhf_dataset":
            model_embeddings_toxic = torch.load(f'{model_representations_directory}/{directory_name}_{badname}_representations_{args.llm_model}_{language}.pt', weights_only=True)
            model_embeddings_benign = torch.load(f'{model_representations_directory}/{directory_name}_benign_representations_{args.llm_model}_{language}.pt', weights_only=True)

        elif args.embeddings_dataset == "llm_lat_dataset":
            model_embeddings_toxic = torch.load(f'{model_representations_directory}/{directory_name}_{badname}_{language}_layer_{args.hidden_states_representations_layer}_representations_{args.llm_model}_{language}.pt', weights_only=True)
            model_embeddings_benign = torch.load(f'{model_representations_directory}/{directory_name}_benign_{language}_layer_{args.hidden_states_representations_layer}_representations_{args.llm_model}_{language}.pt', weights_only=True)

        else:
            if args.use_last_token_embeddings:
                model_embeddings_toxic = torch.load(f'{model_representations_directory}/{directory_name}_{badname}_{language}_last_token_representations_{args.llm_model}_{language}.pt', weights_only=True).float()
                model_embeddings_benign = torch.load(f'{model_representations_directory}/{directory_name}_benign_{language}_last_token_representations_{args.llm_model}_{language}.pt', weights_only=True).float()
            else:
                model_embeddings_toxic = torch.load(f'{model_representations_directory}/{directory_name}_{badname}_{language}_representations_{args.llm_model}_{language}.pt', weights_only=True)
                model_embeddings_benign = torch.load(f'{model_representations_directory}/{directory_name}_benign_{language}_representations_{args.llm_model}_{language}.pt', weights_only=True)

        if len(add_to_all_languages) > 0:
            assert load_separately == True, "we only implemented for this case till now. This is the case for polyguardmix dataset"
            all_languages = all_languages + add_to_all_languages

        if not (args.embeddings_dataset == "polyguardmix_train_100K" or args.embeddings_dataset == "polyguardmix_train_500K" or args.embeddings_dataset == "polyguardmix_train_all" or args.embeddings_dataset == "polyguardmix_train_10K" or args.embeddings_dataset == "llm_lat_dataset"): 
            model_embeddings_toxic = model_embeddings_toxic[:, args.hidden_states_representations_layer, :].clone()
            model_embeddings_benign = model_embeddings_benign[:, args.hidden_states_representations_layer, :].clone()

        if args.pass_through_modifier_model:
            assert args.use_last_token_embeddings
            ## pass the embeddings through the modifier model
            ## load the trained modifier model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            modifier = build_modifier_model(args, hidden_size=model_embeddings_benign.shape[-1])
            if args.custom_train_languages is not None:
                languages_joined = "_".join(args.custom_train_languages)
            else:
                languages_joined = "_".join(args.other_language_translate_to_english)
            modifier_model_path = f"modifier_models_translate_to_english/{args.llm_model}_from_{languages_joined}_to_en_wildguard_filtered_for_length_and_category_epochs_{args.modifier_model_train_epochs}_lr_0.0001_modifier_layers_{args.modifier_model_num_layers}_hidden_layer_{args.hidden_states_representations_layer}.pt"
            if args.remove_translation_prompt:
                ## add "no_trans" to the end of the file name
                modifier_model_path = modifier_model_path.replace(".pt", "_no_trans.pt")
            print(f"loading modifier model from {modifier_model_path}")
            ## it was saved like: torch.save({'epoch': epoch, 'modifier_state_dict': modifier.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),}, save_path), so there is more than just the state dict
            modifier.load_state_dict(torch.load(modifier_model_path, weights_only=True)['modifier_state_dict'])
            modifier.eval()
            with torch.no_grad():
                model_embeddings_toxic = modifier(model_embeddings_toxic)
                model_embeddings_benign = modifier(model_embeddings_benign)
            print(f"modified the embeddings using the modifier model: {model_embeddings_toxic.shape}, {model_embeddings_benign.shape}")

        if args.embeddings_dataset == "toxigen_dataset":
            assert model_embeddings_toxic.shape == torch.Size([1000, args.embedding_dim]) == model_embeddings_benign.shape
        elif args.embeddings_dataset == "jailbreakbench_dataset":
            assert model_embeddings_toxic.shape == torch.Size([100, args.embedding_dim]) == model_embeddings_benign.shape
        elif args.embeddings_dataset == "xstest_dataset":
            assert model_embeddings_toxic.shape == torch.Size([200, args.embedding_dim]) == model_embeddings_benign.shape
        elif args.embeddings_dataset == "aegis_safety_dataset":
            assert model_embeddings_toxic.shape == torch.Size([1268, args.embedding_dim]) == model_embeddings_benign.shape
        elif args.embeddings_dataset == "beavertails_rlhf_dataset":
            assert model_embeddings_toxic.shape == torch.Size([2000, args.embedding_dim]) == model_embeddings_benign.shape
        elif args.embeddings_dataset == "hades_dataset":
            assert model_embeddings_toxic.shape == torch.Size([750, args.embedding_dim]) == model_embeddings_benign.shape
        elif args.embeddings_dataset == "mm_safetybench_dataset":
            assert model_embeddings_toxic.shape == torch.Size([1680, args.embedding_dim]) == model_embeddings_benign.shape, f"model_embeddings_toxic.shape: {model_embeddings_toxic.shape}, model_embeddings_benign.shape: {model_embeddings_benign.shape}"
        elif args.embeddings_dataset == "vlsbench_dataset":
            assert model_embeddings_toxic.shape == torch.Size([2241, args.embedding_dim]) == model_embeddings_benign.shape, f"model_embeddings_toxic.shape: {model_embeddings_toxic.shape}, model_embeddings_benign.shape: {model_embeddings_benign.shape}"
        elif args.embeddings_dataset == "mml_safebench_figstep_dataset":
            assert model_embeddings_toxic.shape == torch.Size([517, args.embedding_dim]) == model_embeddings_benign.shape, f"model_embeddings_toxic.shape: {model_embeddings_toxic.shape}, model_embeddings_benign.shape: {model_embeddings_benign.shape}"
        elif args.embeddings_dataset == "polyguardmix_train_100K":
            assert model_embeddings_toxic.shape == torch.Size([50000, args.embedding_dim]) == model_embeddings_benign.shape, f"model_embeddings_toxic.shape: {model_embeddings_toxic.shape}, model_embeddings_benign.shape: {model_embeddings_benign.shape}"
        elif args.embeddings_dataset == "polyguardmix_train_500K":
            assert model_embeddings_toxic.shape == torch.Size([250000, args.embedding_dim]) == model_embeddings_benign.shape, f"model_embeddings_toxic.shape: {model_embeddings_toxic.shape}, model_embeddings_benign.shape: {model_embeddings_benign.shape}"
        elif args.embeddings_dataset == "polyguardmix_train_all" or args.embeddings_dataset == "polyguardmix_train_10K" or args.embeddings_dataset == "polyguardmix_train_100K" or args.embeddings_dataset == "polyguardmix_train_500K" or args.embeddings_dataset == "llm_lat_dataset" or args.embeddings_dataset == "aegis2_safety_llamaguard_filtered" or args.embeddings_dataset == "aegis2_safety_random_sampled" or args.embeddings_dataset == "wildguard_word_balanced_random_sampled" or args.embeddings_dataset == "original_smaller_llm_lat_dataset" or args.embeddings_dataset == "second_time_aegis_smaller_sampled" or args.embeddings_dataset == "third_time_aegis2_resampled":
            # assert model_embeddings_toxic.shape == torch.Size([500000, args.embedding_dim]) == model_embeddings_benign.shape, f"model_embeddings_toxic.shape: {model_embeddings_toxic.shape}, model_embeddings_benign.shape: {model_embeddings_benign.shape}"
            print(f"model_embeddings_toxic.shape: {model_embeddings_toxic.shape}, model_embeddings_benign.shape: {model_embeddings_benign.shape}")
        else:
            raise NotImplementedError
        
        if split_along_languages:
            if language in train_languages:
                train_embeddings[language] = {}
                train_embeddings[language]['toxic'] = model_embeddings_toxic
                train_embeddings[language]['benign'] = model_embeddings_benign
            elif language in validation_languages:
                validation_embeddings[language] = {}
                validation_embeddings[language]['toxic'] = model_embeddings_toxic
                validation_embeddings[language]['benign'] = model_embeddings_benign
            elif language in test_languages:
                test_embeddings[language] = {}
                test_embeddings[language]['toxic'] = model_embeddings_toxic
                test_embeddings[language]['benign'] = model_embeddings_benign
            else:
                raise NotImplementedError
        
        elif split_along_datapoints:
            train_set_size = 0.7
            validation_set_size = 0.15 + train_set_size
            # test_set_size = 0.25
            ## If this case, we don't split the languages, but split the datapoints into train, validation and test sets. So all languages will go in the train, validation and test sets. Also the harmful and benign datapoints will be merged, we don't need to keep them separate.
            merged_embeddings = torch.cat([model_embeddings_toxic, model_embeddings_benign], dim=0)
            merged_labels = torch.cat([torch.ones(model_embeddings_toxic.shape[0], 1, dtype=torch.int), torch.zeros(model_embeddings_benign.shape[0], 1, dtype=torch.int)], dim=0)
            assert merged_embeddings.shape[0] == merged_labels.shape[0] == num_datapoints_total == len(shuffled_order_datapoints)
            ## split the datapoints into train, validation and test sets: 60% train, 15% validation, 25% test (after shuffling using the using deterministic_shuffle function with the random_seed)
            merged_embeddings = merged_embeddings[shuffled_order_datapoints]
            merged_labels = merged_labels[shuffled_order_datapoints].squeeze()
            train_embeddings_all_this_language = merged_embeddings[:int(train_set_size*merged_embeddings.shape[0])]
            validation_embeddings_all_this_language = merged_embeddings[int(train_set_size*merged_embeddings.shape[0]):int(validation_set_size*merged_embeddings.shape[0])]
            test_embeddings_all_this_language = merged_embeddings[int(validation_set_size*merged_embeddings.shape[0]):]
            train_labels_all_this_language = merged_labels[:int(train_set_size*merged_labels.shape[0])]
            validation_labels_all_this_language = merged_labels[int(train_set_size*merged_labels.shape[0]):int(validation_set_size*merged_labels.shape[0])]
            test_labels_all_this_language = merged_labels[int(validation_set_size*merged_labels.shape[0]):]
            train_embeddings[language] = {}
            train_embeddings[language]['toxic'] = train_embeddings_all_this_language[train_labels_all_this_language == 1]
            train_embeddings[language]['benign'] = train_embeddings_all_this_language[train_labels_all_this_language == 0]
            validation_embeddings[language] = {}
            validation_embeddings[language]['toxic'] = validation_embeddings_all_this_language[validation_labels_all_this_language == 1]
            validation_embeddings[language]['benign'] = validation_embeddings_all_this_language[validation_labels_all_this_language == 0]
            test_embeddings[language] = {}
            test_embeddings[language]['toxic'] = test_embeddings_all_this_language[test_labels_all_this_language == 1]
            test_embeddings[language]['benign'] = test_embeddings_all_this_language[test_labels_all_this_language == 0]
            print(f"Using {train_embeddings_all_this_language.shape[0]} train, {validation_embeddings_all_this_language.shape[0]} validation and {test_embeddings_all_this_language.shape[0]} test embeddings for {language} language.")
        
        elif split_along_languages_and_datapoints:
            ## here we will first split dataset into 70% train, 15% validation, 15% test. Then we will split the languages into train, validation and test sets. So 70% of some languages will go in the train set, 15% of some other languages will go in the validation set, and 15% of the remaining languages will go in the test set.
            merged_embeddings = torch.cat([model_embeddings_toxic, model_embeddings_benign], dim=0)
            merged_labels = torch.cat([torch.ones(model_embeddings_toxic.shape[0], 1, dtype=torch.int), torch.zeros(model_embeddings_benign.shape[0], 1, dtype=torch.int)], dim=0)
            assert merged_embeddings.shape[0] == merged_labels.shape[0] == num_datapoints_total == len(shuffled_order_datapoints)
            shuffled_embeddings = merged_embeddings[shuffled_order_datapoints]
            shuffled_labels = merged_labels[shuffled_order_datapoints].squeeze()

            if args.split_train_test_along_languages_and_datapoints:
                train_set_size = 0.7
                validation_set_size = 0.15 + train_set_size
                test_set_size = 0.15
            elif args.split_train_test_along_languages_and_datapoints_1_train:
                train_set_size = 0.01
                validation_set_size = 0.2 + train_set_size
                test_set_size = 0.79
            elif args.split_train_test_along_languages_and_datapoints_2_train:
                train_set_size = 0.02
                validation_set_size = 0.2 + train_set_size
                test_set_size = 0.78
            elif args.split_train_test_along_languages_and_datapoints_5_train:
                train_set_size = 0.05
                validation_set_size = 0.2 + train_set_size
                test_set_size = 0.75
            elif args.split_train_test_along_languages_and_datapoints_10_train:
                train_set_size = 0.1
                validation_set_size = 0.2 + train_set_size
                test_set_size = 0.7
            elif args.split_train_test_along_languages_and_datapoints_20_train:
                train_set_size = 0.2
                validation_set_size = 0.2 + train_set_size
                test_set_size = 0.6
            elif args.split_train_test_along_languages_and_datapoints_30_train:
                train_set_size = 0.3
                validation_set_size = 0.2 + train_set_size
                test_set_size = 0.5
            elif args.split_train_test_along_languages_and_datapoints_40_train:
                train_set_size = 0.4
                validation_set_size = 0.2 + train_set_size
                test_set_size = 0.4
            elif args.split_train_test_along_languages_and_datapoints_50_train:
                train_set_size = 0.5
                validation_set_size = 0.2 + train_set_size
                test_set_size = 0.3
            elif args.split_train_test_along_languages_and_datapoints_60_train:
                train_set_size = 0.6
                validation_set_size = 0.2 + train_set_size
            elif args.split_train_test_along_languages_and_datapoints_50_datapoints:
                train_set_size = '50_datapoints'
                validation_set_size = 0.2
            elif args.split_train_test_along_languages_and_datapoints_100_datapoints:
                train_set_size = '100_datapoints'
                validation_set_size = 0.2
            else:
                raise NotImplementedError

            if language in train_languages:
                train_embeddings[language] = {}
                if train_set_size == '50_datapoints':
                    final_embeddings_to_consider = shuffled_embeddings[:50]
                    final_labels_to_consider = shuffled_labels[:50]
                elif train_set_size == '100_datapoints':
                    final_embeddings_to_consider = shuffled_embeddings[:100]
                    final_labels_to_consider = shuffled_labels[:100]
                else:
                    final_embeddings_to_consider = shuffled_embeddings[:int(train_set_size*shuffled_embeddings.shape[0])]
                    final_labels_to_consider = shuffled_labels[:int(train_set_size*shuffled_embeddings.shape[0])]
                
                train_embeddings[language]['toxic'] = final_embeddings_to_consider[final_labels_to_consider == 1]     ## the toxic ones r the ones whose labels are 1
                train_embeddings[language]['benign'] = final_embeddings_to_consider[final_labels_to_consider == 0]
                if train_set_size == '50_datapoints' or train_set_size == '100_datapoints':
                    print(f"Using 50 or 100 datapoints of the {language} datapoints for training.")
                else:
                    print(f"Using {train_set_size*100}% of the {language} datapoints for training.")
            
            elif language in validation_languages:
                validation_embeddings[language] = {}
                if train_set_size == '50_datapoints':
                    final_embeddings_to_consider = shuffled_embeddings[50:int(validation_set_size*shuffled_embeddings.shape[0]) + 50]
                    final_labels_to_consider = shuffled_labels[50:int(validation_set_size*shuffled_embeddings.shape[0]) + 50]
                elif train_set_size == '100_datapoints':
                    final_embeddings_to_consider = shuffled_embeddings[100:int(validation_set_size*shuffled_embeddings.shape[0]) + 100]
                    final_labels_to_consider = shuffled_labels[100:int(validation_set_size*shuffled_embeddings.shape[0]) + 100]
                else:
                    final_embeddings_to_consider = shuffled_embeddings[int(train_set_size*shuffled_embeddings.shape[0]):int(validation_set_size*shuffled_embeddings.shape[0])]
                    final_labels_to_consider = shuffled_labels[int(train_set_size*shuffled_embeddings.shape[0]):int(validation_set_size*shuffled_embeddings.shape[0])]
            
                validation_embeddings[language]['toxic'] = final_embeddings_to_consider[final_labels_to_consider == 1]
                validation_embeddings[language]['benign'] = final_embeddings_to_consider[final_labels_to_consider == 0]
                if train_set_size == '50_datapoints' or train_set_size == '100_datapoints':
                    print(f"Using {(validation_set_size) * 100}% of the {language} datapoints for validation.")
                else:
                    print(f"Using {(validation_set_size - train_set_size) * 100}% of the {language} datapoints for validation.")
            
            elif language in test_languages:
                test_embeddings[language] = {}
                if train_set_size == '50_datapoints':
                    final_embeddings_to_consider = shuffled_embeddings[50 + int(validation_set_size*shuffled_embeddings.shape[0]):]
                    final_labels_to_consider = shuffled_labels[50 + int(validation_set_size*shuffled_embeddings.shape[0]):]
                elif train_set_size == '100_datapoints':
                    final_embeddings_to_consider = shuffled_embeddings[100 + int(validation_set_size*shuffled_embeddings.shape[0]):]
                    final_labels_to_consider = shuffled_labels[100 + int(validation_set_size*shuffled_embeddings.shape[0]):]
                else:
                    final_embeddings_to_consider = shuffled_embeddings[int(validation_set_size*shuffled_embeddings.shape[0]):]
                    final_labels_to_consider = shuffled_labels[int(validation_set_size*shuffled_embeddings.shape[0]):]
            
                test_embeddings[language]['toxic'] = final_embeddings_to_consider[final_labels_to_consider == 1]
                test_embeddings[language]['benign'] = final_embeddings_to_consider[final_labels_to_consider == 0]
                if train_set_size == '50_datapoints' or train_set_size == '100_datapoints':
                    print(f"Using {(1 - validation_set_size) * 100}% of the {language} datapoints for testing.")
                else:
                    print(f"Using {(1 - validation_set_size) * 100}% of the {language} datapoints for testing.")
            
            else:
                raise NotImplementedError

        elif load_separately:
            pass
        
        else:
            raise NotImplementedError
        
        print(f"Loaded {language} embeddings successfully!")
    
    return train_embeddings, validation_embeddings, test_embeddings


def prepare_data_loaders_for_harmfulness(args, this_embeddings, each_language_separate, add_language_labels=False, languages_to_consider=None, subtract_CS_LRD_component=None, train_set=True, trained_eraser=None, modify_using_LEACE=False):
    if each_language_separate:
        collect_data_here = {}
        collect_labels_here = {}
        data_loaders_all_languages_separately = {}
    else:
        all_data = []
        all_harmfulness_labels = []
        language_labels_all = []
        
    for here_language in this_embeddings:
        if 'toxic' in this_embeddings[here_language]:
            toxic_embeddings = this_embeddings[here_language]['toxic']
            toxic_labels = torch.ones(toxic_embeddings.shape[0], 1)
        else:
            ## generate empty tensor that can be concatenated later, but has no effect
            toxic_embeddings, toxic_labels = torch.empty((0, args.embedding_dim)), torch.empty((0, 1))
        if 'benign' in this_embeddings[here_language]:
            benign_embeddings = this_embeddings[here_language]['benign']
            benign_labels = torch.zeros(benign_embeddings.shape[0], 1)
        else:
            benign_embeddings, benign_labels = torch.empty((0, args.embedding_dim)), torch.empty((0, 1))
        
        if subtract_CS_LRD_component is not None:
            toxic_embeddings = toxic_embeddings - torch.matmul(torch.matmul(subtract_CS_LRD_component, subtract_CS_LRD_component.T), toxic_embeddings.T).T
            benign_embeddings = benign_embeddings - torch.matmul(torch.matmul(subtract_CS_LRD_component, subtract_CS_LRD_component.T), benign_embeddings.T).T
            
        if each_language_separate:
            collect_data_here[here_language] = torch.cat([toxic_embeddings, benign_embeddings], dim=0)
            collect_labels_here[here_language] =  torch.cat([toxic_labels, benign_labels], dim=0)
            if modify_using_LEACE and train_set:
                assert NotImplementedError      ## this cannot happen as for training the leace eraser, we need the entire dataset for all languages.
            elif modify_using_LEACE and not train_set: 
                assert trained_eraser is not None
                collect_data_here[here_language] = trained_eraser(collect_data_here[here_language].cuda()).cpu()
                print("Modified the embeddings using LEACE for the test set ")
                
            this_language_dataset = TensorDataset(collect_data_here[here_language], collect_labels_here[here_language])
            data_loaders_all_languages_separately[here_language] = DataLoader(this_language_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        else:
            all_data.append(torch.cat([toxic_embeddings, benign_embeddings], dim=0))
            all_harmfulness_labels.append(torch.cat([toxic_labels, benign_labels], dim=0))
            if add_language_labels:
                language_label_here = torch.tensor([languages_to_consider.index(here_language)] * (toxic_embeddings.shape[0] + benign_embeddings.shape[0])).unsqueeze(1)
                language_labels_all.append(language_label_here)

    if each_language_separate:
        return collect_data_here, collect_labels_here, data_loaders_all_languages_separately, None
    else:
        all_data = torch.cat(all_data, dim=0)
        all_harmfulness_labels = torch.cat(all_harmfulness_labels, dim=0)
        if add_language_labels:
            language_labels_all = torch.cat(language_labels_all, dim=0)
            if modify_using_LEACE and train_set:
                from concept_erasure import LeaceEraser
                print("Modifying the embeddings using LEACE 1")
                trained_eraser = LeaceEraser.fit(all_data.cuda(), torch.nn.functional.one_hot(language_labels_all, num_classes=language_labels_all.max().item() + 1).squeeze().cuda())
                all_data_new = trained_eraser(all_data.cuda()).cpu()
                print("Modified the embeddings using LEACE for the train set ")
                this_embeddings_dataset = TensorDataset(all_data_new, all_harmfulness_labels, language_labels_all)
            elif modify_using_LEACE and not train_set:
                assert trained_eraser is not None
                all_data_new = trained_eraser(all_data.cuda()).cpu()
                print("Modified the embeddings using LEACE for the test set ")
                this_embeddings_dataset = TensorDataset(all_data_new, all_harmfulness_labels, language_labels_all)
            else:
                this_embeddings_dataset = TensorDataset(all_data, all_harmfulness_labels, language_labels_all)
        else:
            this_embeddings_dataset = TensorDataset(all_data, all_harmfulness_labels)
        
        data_loaders_this_embeddings = DataLoader(this_embeddings_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
        return all_data, all_harmfulness_labels, data_loaders_this_embeddings, trained_eraser


def create_classifier_harmfulness(args, num_neurons):
    ## given this list of neurons, create a NN with them with relu in between. It will start with args.embedding_dim and end with 1 neuron.
    if num_neurons is None:
        num_neurons = [2048, 1024, 512, 256, 128]
    
    classifier_model = []
    input_dim = args.embedding_dim
    for i in range(len(num_neurons)):
        classifier_model.append(nn.Linear(input_dim, num_neurons[i]))
        classifier_model.append(nn.ReLU())
        input_dim = num_neurons[i]
    classifier_model.append(nn.Linear(input_dim, 1))
    classifier_model = nn.Sequential(*classifier_model)
    print(f"Classifier model: {classifier_model}")
    return classifier_model


def train_mlp_classifier(args, train_loader, train_languages_used, data_loaders_all_validation_languages_separately, data_loaders_all_test_languages_separately, project_name=None, run_name=None, only_print_best_accuracy=False, num_neurons=None, all_external_test_data_loaders=None):
    classifier_model = create_classifier_harmfulness(args, num_neurons)
    # initialize the weights of the classifier model with xavier initialization
    for layer in classifier_model:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    criterion_classifier = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(classifier_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
    classifier_model = classifier_model.cuda()
    if args.measure_inference_time:
        assert args.batch_size == 1, "Inference time measurement is only supported for batch size 1."
        args.num_epochs = 1
    
    if args.log_to_wandb:
        wandb_logging = init_wandb_and_log(args, project_name=project_name, run_name=run_name)
    
    ## TRAINING THE CLASSIFIER. AT THE END OF EVERY 10 EPOCHS, MEASURE THE VALIDATION AND TEST ACCURACIES.
    if all_external_test_data_loaders is not None:
        import json
        output_results_file = f"external_test_results/results_{args.llm_model}_{args.embeddings_dataset}_lyr_{args.hidden_states_representations_layer}_tr_{'_'.join(train_languages_used)}_lr_{args.learning_rate}_bsz_{args.batch_size}.jsonl"
        open(output_results_file, "w").close()
    print("Starting training the classifier...")
    best_validation_accuracy = 0
    best_test_accuracy = 0
    test_accuracies_each_language_for_best_validation_accuracy = {}
    for epoch in range(args.num_epochs):
        classifier_model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.cuda(), labels.cuda()
            optimizer.zero_grad()
            output = classifier_model(data)
            loss = criterion_classifier(output, labels)
            # if epoch > 0:
            if not args.measure_inference_time:     ## we will just forward pass once through the untrained model. 
                loss.backward()
                optimizer.step()
        
        if (epoch % args.log_epochs == 0 or epoch == args.num_epochs - 1):
            def eval_model_on_data(data_loaders_all_languages_separately):
                classifier_model.eval()
                with torch.no_grad():
                    accuracy_this_language = {}
                    for language in data_loaders_all_languages_separately:
                        predictions = []
                        true_labels = []
                        
                        if args.measure_inference_time:
                            gpu_batches = []
                            for data, _ in data_loaders_all_languages_separately[language]:
                                gpu_batches.append(data.cuda(non_blocking=True))
                            print(f"number of batches: {len(gpu_batches)}")
                            start_time = time.time()
                            for data in gpu_batches:
                                output = classifier_model(data)
                                predictions.append(torch.sigmoid(output) > 0.5)
                            end_time = time.time()
                            print(f"End time: {end_time}")
                            elapsed_time = end_time - start_time
                            print(f"\033[93m Elapsed time for {language}: {elapsed_time} seconds \033[0m")
                            exit()

                        for batch_idx, (data_here, labels_here) in enumerate(data_loaders_all_languages_separately[language]):
                            data_here, labels_here = data_here.cuda(), labels_here.cuda()
                            output = classifier_model(data_here)
                            predictions.append(torch.sigmoid(output) > 0.5)
                            true_labels.append(labels_here)
                        predictions = torch.cat(predictions, dim=0)
                        true_labels = torch.cat(true_labels, dim=0)
                        accuracy = (predictions == true_labels).float().mean().item()
                        accuracy_this_language[language] = round(accuracy * 100, 1)
                return accuracy_this_language
            
            if args.measure_inference_time:
                external_test_accuracy = eval_model_on_data(all_external_test_data_loaders)
                continue

            test_accuracy = eval_model_on_data(data_loaders_all_test_languages_separately)
            if data_loaders_all_validation_languages_separately is not None:
                validation_accuracy = eval_model_on_data(data_loaders_all_validation_languages_separately)
            else:
                validation_accuracy = test_accuracy
            if all_external_test_data_loaders is not None:
                external_test_accuracy = eval_model_on_data(all_external_test_data_loaders)
            
            mean_nat_lang_test_accuracy = round(np.mean([test_accuracy[lang] for lang in natural_languages if lang in test_accuracy]), 1)
            mean_cipher_lang_test_accuracy = round(np.mean([test_accuracy[lang] for lang in cipher_languages if lang in test_accuracy]), 1)
            mean_nat_lang_validation_accuracy = round(np.mean([validation_accuracy[lang] for lang in natural_languages if lang in validation_accuracy]), 1)
            mean_cipher_lang_validation_accuracy = round(np.mean([validation_accuracy[lang] for lang in cipher_languages if lang in validation_accuracy]), 1)
            if all_external_test_data_loaders is not None:
                mean_external_test_accuracy = round(np.mean([external_test_accuracy[lang] for lang in all_external_test_data_loaders]), 1)
                epoch_metrics = {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    # ── MAIN (“in-domain”) DATASET ────────────────────────────────
                    "main": {
                        "test_accuracy":            test_accuracy,              # lang → acc
                        "validation_accuracy":      validation_accuracy,        # lang → acc
                        "mean_nat_test_accuracy":   mean_nat_lang_test_accuracy,
                        "mean_cipher_test_accuracy":mean_cipher_lang_test_accuracy,
                        "mean_nat_val_accuracy":    mean_nat_lang_validation_accuracy,
                        "mean_cipher_val_accuracy": mean_cipher_lang_validation_accuracy,
                    },
                }
                # ── EXTERNAL DATASETS (each becomes its own top-level key) ──────
                for ext_ds in args.load_separate_test_set_name:
                    ext_acc = {k: v for k, v in external_test_accuracy.items() if k.startswith(ext_ds)}
                    epoch_metrics[ext_ds] = {
                        "test_accuracy":  ext_acc,                         # lang / subset → acc
                        "mean_accuracy":  round(np.mean(list(ext_acc.values())), 2),
                    }

                # overall average across *all* external measurements
                epoch_metrics["external_overall_mean_accuracy"] = mean_external_test_accuracy
                ## write the file at each of each epoch
                import io
                with open(output_results_file, "a") as f:
                    f.write(json.dumps(epoch_metrics) + "\n")
            
            if not only_print_best_accuracy:
                if all_external_test_data_loaders is not None:
                    print(f"Epoch: {epoch}, \033[93mMean Natural Val Acc.: {mean_nat_lang_validation_accuracy}, Mean Cipher Val Acc.: {mean_cipher_lang_validation_accuracy}, Mean Natural Test Acc.: {mean_nat_lang_test_accuracy}, Mean Cipher Test Acc.: {mean_cipher_lang_test_accuracy}, Mean External Test Accuracy: {mean_external_test_accuracy} lr: {optimizer.param_groups[0]['lr']}\033[0m, Test Accuracy: {test_accuracy}, Validation Accuracy: {validation_accuracy}")
                    ## for external test_accuracies we will print the avg for each dataset separately
                    for external_test_set in args.load_separate_test_set_name:
                        ## find all the keys in the external_test_accuracy that start with the external_test_set
                        external_test_set_accuracy = {k: v for k, v in external_test_accuracy.items() if k.startswith(external_test_set)}
                        mean_external_test_set_accuracy = round(np.mean([external_test_set_accuracy[lang] for lang in external_test_set_accuracy]), 1)
                        print(f"\033[93mExternal Test Set: {external_test_set}, Mean Accuracy: {mean_external_test_set_accuracy}\033[0m, Test Accuracy: {external_test_set_accuracy}")
                else:    
                    print(f"Epoch: {epoch}, \033[93mMean Natural Val Acc.: {mean_nat_lang_validation_accuracy}, Mean Cipher Val Acc.: {mean_cipher_lang_validation_accuracy}, Mean Natural Test Acc.: {mean_nat_lang_test_accuracy}, Mean Cipher Test Acc.: {mean_cipher_lang_test_accuracy}, lr: {optimizer.param_groups[0]['lr']}\033[0m, Test Accuracy: {test_accuracy}, Validation Accuracy: {validation_accuracy}")
            
            # mean_validation_accuracy = sum(list(validation_accuracy.values()))/len(validation_accuracy)
            ## if mean cipher accuracy is not nan, then use that as the validation accuracy, else use the mean natural language validation accuracy
            if np.isnan(mean_cipher_lang_validation_accuracy):
                mean_val_accuracy_to_consider = mean_nat_lang_validation_accuracy
            else:
                mean_val_accuracy_to_consider = mean_cipher_lang_validation_accuracy
            
            if mean_val_accuracy_to_consider > best_validation_accuracy:
                best_validation_accuracy = mean_val_accuracy_to_consider
                if np.isnan(mean_cipher_lang_test_accuracy):
                    best_test_accuracy = mean_nat_lang_test_accuracy
                else:
                    best_test_accuracy = mean_cipher_lang_test_accuracy
                ## copy the test accuracy to variable: test_accuracies_each_language_for_best_validation_accuracy
                test_accuracies_each_language_for_best_validation_accuracy = test_accuracy.copy()
                ## save the model if asked for
                if args.save_best_harmfulness_model:
                    os.makedirs("best_harmfulness_classifiers", exist_ok=True)
                    model_save_path = f"best_harmfulness_classifiers/{args.llm_model}_{args.embeddings_dataset}_lyr_{args.hidden_states_representations_layer}_tr_{'_'.join(train_languages_used)}_lr_{args.learning_rate}_bsz_{args.batch_size}.pt"    #_seed_{args.random_seed_for_datapoint_split}_mdl_{'_'.join([str(x) for x in num_neurons])}.pt"
                    ## save the model so that it can be loaded later for inference
                    torch.save({
                        'epoch': epoch,
                        'classifier_state_dict': classifier_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, model_save_path)
                    print(f"Saved the model to {model_save_path}")

            scheduler.step(np.mean(list(validation_accuracy.values())))
            
            if args.log_to_wandb:
                # wandb_logging.log({"validation_accuracy": sum(list(validation_accuracy.values()))/len(validation_accuracy), "test_accuracy": sum(list(test_accuracy.values()))/len(test_accuracy), "mean_nat_lang_test_accuracy": mean_nat_lang_test_accuracy, "mean_cipher_lang_test_accuracy": mean_cipher_lang_test_accuracy, "epoch": epoch})
                wandb_logging.log(
                    {
                        "natural_language_validation_accuracy": mean_nat_lang_validation_accuracy,
                        "cipher_language_validation_accuracy": mean_cipher_lang_validation_accuracy,
                        "natural_language_test_accuracy": mean_nat_lang_test_accuracy,
                        "cipher_language_test_accuracy": mean_cipher_lang_test_accuracy,
                        "external_test_accuracy": mean_external_test_accuracy if all_external_test_data_loaders is not None else None,
                        "epoch": epoch,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        ## also log the individual test accuracies for each language -- there are many languages, so would the key for wandb change per langauges?
                        "test_accuracy": test_accuracy,
                        "validation_accuracy": validation_accuracy,
                        "mean_external_test_accuracy": mean_external_test_accuracy if all_external_test_data_loaders is not None else None,
                        "external_test_accuracy": external_test_accuracy if all_external_test_data_loaders is not None else None,
                    }
                )
            
    print("Training done!")
    # print(f"Epoch: {epoch}, Validation Accuracy: {validation_accuracy}, Test Accuracy: {test_accuracy}, \033[93mMean Validation Accuracy: {round(sum(list(validation_accuracy.values()))/len(validation_accuracy), 1)}, Mean Test Accuracy: {round(sum(list(test_accuracy.values()))/len(test_accuracy), 1)}, Mean Natural Language Test Accuracy: {mean_nat_lang_test_accuracy}, Mean Cipher Language Test Accuracy: {mean_cipher_lang_test_accuracy}, lr: {optimizer.param_groups[0]['lr']}\033[0m")
    print(f"Best Validation Accuracy: {best_validation_accuracy}, Best Test Accuracy: {best_test_accuracy}, Test Accuracies for each language for best validation accuracy: {test_accuracies_each_language_for_best_validation_accuracy}")
    if args.split_only_cipher_languages_randomly and args.train_classifier_harmfulness:
        ## print the test accuracies from the best epoch with the random seed for splitting the languages to an output file. 
        if args.split_train_test_along_only_datapoints:
            output_file = "test_accuracies_for_best_validation_accuracy_cipher_languages_only_datapoints.txt"
        else:
            output_file = "test_accuracies_for_best_validation_accuracy_cipher_languages.txt"
        with open(output_file, "a") as f:
            f.write(f"random_seed_for_language_split: {args.random_seed_for_language_split}, Test Accuracies for each language for best validation accuracy: {test_accuracies_each_language_for_best_validation_accuracy}\n")

    print(f"Best Validation Accuracy: {best_validation_accuracy}, Best Test Accuracy: {best_test_accuracy}")
    
    if args.finetune_on_external_datasets:
        output_finetuning_file = f"finetuning_test_results/multiguard_results/results_{args.llm_model}_{args.embeddings_dataset}_lr_{args.learning_rate}_bsz_{args.batch_size}_num_shots_{args.num_few_shot_examples}_random_seed_{args.random_seed_for_datapoint_split}.jsonl"
        assert args.external_datasets_for_finetuning is not None, "Please provide the external datasets for finetuning."
        ## if this is true, we will load the embeddings of the finetuning external datasets and finetune the model on them -- probably lower learning rate I guess. 
        finetuning_embeddings = {}
        for external_dataset in args.external_datasets_for_finetuning:
            print(f"Finetuning on {external_dataset}...")
            external_finetuning_embeddings = load_external_embeddings_for_finetuning(args, external_dataset)
            finetuning_embeddings.update(external_finetuning_embeddings)
        ## now create the data loaders for the finetuning embeddings
        if args.num_few_shot_examples > 0:
            _, _, train_loader, _ = prepare_data_loaders_for_harmfulness(args, finetuning_embeddings, each_language_separate=False)
            ## now finetune the model on the finetuning embeddings
            print(f"Finetuning the model on {args.external_datasets_for_finetuning}...")
            ## create a optimizer with a lower learning rate
            optimizer_ft = optim.AdamW(classifier_model.parameters(), lr=args.learning_rate / 10)
            # scheduler_ft = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='max', factor=0.5, patience=4)
            classifier_model.train()
        for epoch_ft in range(100):
            if args.num_few_shot_examples > 0:
                for batch_idx, (data_ft, labels_ft) in enumerate(train_loader):
                    data_ft, labels_ft = data_ft.cuda(), labels_ft.cuda()
                    optimizer_ft.zero_grad()
                    output_ft = classifier_model(data_ft)
                    loss_ft = criterion_classifier(output_ft, labels_ft)
                    loss_ft.backward()
                    optimizer_ft.step()
                    print(f"finetuning batch {batch_idx}, loss: {loss_ft.item()}")
            ## now evaluate the model on the test set
            if epoch_ft % 1 == 0:
                print(f"Evaluating model at epoch {epoch_ft}...")
                test_accuracy = eval_model_on_data(data_loaders_all_test_languages_separately)
                mean_nat_lang_test_accuracy = round(np.mean([test_accuracy[lang] for lang in natural_languages if lang in test_accuracy]), 1)
                if args.num_few_shot_examples > 0:
                    print(f"Finetuning Epoch: {epoch_ft}, \033[93m Mean Natural Test Acc.: {mean_nat_lang_test_accuracy}, lr: {optimizer_ft.param_groups[0]['lr']}\033[0m, Test Accuracy: {test_accuracy}")
                else:
                    print(f"Finetuning Epoch: {epoch_ft}, \033[93m Mean Natural Test Acc.: {mean_nat_lang_test_accuracy}, \033[0m, Test Accuracy: {test_accuracy}")
                external_test_accuracy = eval_model_on_data(all_external_test_data_loaders)
                for external_test_set in args.load_separate_test_set_name:
                    ## find all the keys in the external_test_accuracy that start with the external_test_set
                    external_test_set_accuracy = {k: v for k, v in external_test_accuracy.items() if k.startswith(external_test_set)}
                    mean_external_test_set_accuracy = round(np.mean([external_test_set_accuracy[lang] for lang in external_test_set_accuracy]), 1)
                    print(f"\033[93mExternal Test Set: {external_test_set}, Mean Accuracy: {mean_external_test_set_accuracy}\033[0m, Test Accuracy: {external_test_set_accuracy}")
                
                epoch_ft_metrics = {
                    "ft epoch": epoch_ft,
                    "main": {
                        "test_accuracy":            test_accuracy,              # lang → acc
                        "num_few_shot_examples":   args.num_few_shot_examples,
                    },
                }
                for ext_ds in args.load_separate_test_set_name:
                    ext_acc = {k: v for k, v in external_test_accuracy.items() if k.startswith(ext_ds)}
                    epoch_ft_metrics[ext_ds] = {
                        "test_accuracy":  ext_acc,                         # lang / subset → acc
                        "mean_accuracy":  round(np.mean(list(ext_acc.values())), 2),
                    }

                ## now store the output of the finetuning in a file
                with open(output_finetuning_file, "a") as f:
                    f.write(json.dumps(epoch_ft_metrics) + "\n")
                if args.num_few_shot_examples == 0:
                    exit()
        exit()

    ## print the results to a file:
    with open("results.txt", "a") as f:
        f.write(f"{args.custom_train_languages}, {args.custom_test_languages}, {args.modifier_model_num_layers}, {args.random_seed_for_datapoint_split}\n")
        f.write(f"Best Validation Accuracy: {best_validation_accuracy}, Best Test Accuracy: {best_test_accuracy}, Test Accuracies for each language for best validation accuracy: {test_accuracies_each_language_for_best_validation_accuracy}\n\n")
    
    if args.split_only_cipher_languages_randomly:
        return test_accuracies_each_language_for_best_validation_accuracy
    
    return best_test_accuracy


def load_external_embeddings_for_finetuning(args, load_external_dataset):
    if load_external_dataset in ['code_attack_python_list', 'code_attack_python_stack', 'code_attack_python_string']:
        model_embeddings = torch.load(f"CodeAttack/harmful_prompts/model_representations_multilingual_jailbreaks/{load_external_dataset}_code_representations_{args.llm_model}_code.pt")
    elif load_external_dataset in ['code_attack_python_list_benign', 'code_attack_python_stack_benign', 'code_attack_python_string_benign']:
        # model_embeddings = torch.load(f"CodeAttack/benign_prompts/model_representations_multilingual_jailbreaks/{load_external_dataset}_code_representations_{args.llm_model}_code.pt")
        model_embeddings = torch.load(f"CodeAttack/test_benign_prompts/model_representations_multilingual_jailbreaks/{load_external_dataset}_testset_code_representations_{args.llm_model}_code.pt")
    else:
        raise NotImplementedError
    model_embeddings = model_embeddings[:, args.hidden_states_representations_layer, :].clone()
    ## randomly select 10 embeddings from the model_embeddings
    ## seed torch seed using: random_seed_for_datapoint_split
    torch.manual_seed(args.random_seed_for_datapoint_split)
    random_indices = torch.randperm(model_embeddings.shape[0])[:args.num_few_shot_examples]
    model_embeddings = model_embeddings[random_indices, :].clone()

    finetune_embeddings = {}
    finetune_embeddings[load_external_dataset + "_code"] = {}
    if load_external_dataset in ['code_attack_python_list', 'code_attack_python_stack', 'code_attack_python_string']:
        finetune_embeddings[load_external_dataset + "_code"]['toxic'] = model_embeddings
    elif load_external_dataset in ['code_attack_python_list_benign', 'code_attack_python_stack_benign', 'code_attack_python_string_benign']:
        finetune_embeddings[load_external_dataset + "_code"]['benign'] = model_embeddings
    else:
        raise NotImplementedError
    print(f"Loaded {load_external_dataset} embeddings successfully!")
    return finetune_embeddings


def train_vanilla_classifier_on_embeddings_to_predict_harmfulness(args):
    global all_languages, cipher_languages, natural_languages

    if args.split_train_test_along_only_languages:
        train_embeddings, validation_embeddings, test_embeddings = load_embeddings(args, split_along_languages=True, split_along_datapoints=False, split_along_languages_and_datapoints=False)
    elif args.split_train_test_along_languages_and_datapoints or args.split_train_test_along_languages_and_datapoints_1_train or args.split_train_test_along_languages_and_datapoints_2_train or args.split_train_test_along_languages_and_datapoints_5_train or args.split_train_test_along_languages_and_datapoints_10_train or args.split_train_test_along_languages_and_datapoints_40_train or args.split_train_test_along_languages_and_datapoints_20_train or args.split_train_test_along_languages_and_datapoints_30_train or args.split_train_test_along_languages_and_datapoints_50_train or args.split_train_test_along_languages_and_datapoints_60_train or args.split_train_test_along_languages_and_datapoints_50_datapoints or args.split_train_test_along_languages_and_datapoints_100_datapoints:
        train_embeddings, validation_embeddings, test_embeddings = load_embeddings(args, split_along_languages=False, split_along_datapoints=False, split_along_languages_and_datapoints=True)
    elif args.split_train_test_along_only_datapoints:
        train_embeddings, validation_embeddings, test_embeddings = load_embeddings(args, split_along_languages=False, split_along_datapoints=True, split_along_languages_and_datapoints=False)
    elif args.train_test_already_split_load_separately:
        train_embeddings, validation_embeddings, test_embeddings = load_embeddings(args, split_along_languages=False, split_along_datapoints=False, split_along_languages_and_datapoints=False, load_separately=True)
    elif args.split_train_test_along_datapoints_and_use_separate_test_set:      ## this is the case when we want to see test loss also on external datasets
        train_embeddings, validation_embeddings, test_embeddings = load_embeddings(args, split_along_languages=False, split_along_datapoints=True, split_along_languages_and_datapoints=False)
        ## here we will add the embeddings for the external dataset to the test embeddings
        ## merge all external embeddings in test_embeddings (both are dicts), but before that assert that no keys in more test embeddings match the keys in test embeddings
        all_external_test_embeddings = {}
        for external_test_set in args.load_separate_test_set_name: ## this is a list of test set
            external_test_embeddings = load_embeddings_for_testing(args, external_test_set)
            for key_language in external_test_embeddings:
                all_external_test_embeddings[key_language] = external_test_embeddings[key_language]
    else:
        raise NotImplementedError

    print(f"train_embeddings:", train_embeddings.keys())
    print(f"validation_embeddings:", validation_embeddings.keys())
    print(f"test_embeddings:", test_embeddings.keys())
    print(f"all_external_test_embeddings:", all_external_test_embeddings.keys())

    assert sorted(list(set(list(train_embeddings.keys()) + list(validation_embeddings.keys()) + list(test_embeddings.keys())))) == sorted(all_languages)
    ## load the train embeddings, all combined in one dataset
    train_data, train_labels, train_loader, _ = prepare_data_loaders_for_harmfulness(args, train_embeddings, each_language_separate=False)

    ## load the test and validation embeddings, each language separately
    test_data, test_labels, data_loaders_all_test_languages_separately, _ = prepare_data_loaders_for_harmfulness(args, test_embeddings, each_language_separate=False if args.train_classifier_harmfulness_logistic else True)
    if args.split_train_test_along_datapoints_and_use_separate_test_set:
        external_test_data, external_test_labels, data_loaders_all_external_test_languages_separately, _ = prepare_data_loaders_for_harmfulness(args, all_external_test_embeddings, each_language_separate=False if args.train_classifier_harmfulness_logistic else True)
    if args.embeddings_dataset == "polyguardmix_train_100K" or args.embeddings_dataset == "polyguardmix_train_500K" or args.embeddings_dataset == "polyguardmix_train_all" or args.embeddings_dataset == "polyguardmix_train_10K":
        data_loaders_all_validation_languages_separately = None
    else:
        validation_data, validation_labels, data_loaders_all_validation_languages_separately, _ = prepare_data_loaders_for_harmfulness(args, validation_embeddings, each_language_separate=False if args.train_classifier_harmfulness_logistic else True)
            
    if args.train_classifier_harmfulness_logistic:
        print("Training a logistic regression classifier on the embeddings to predict the harmfulness of the language.")
        ## here we will just train a simple logistic regression classifier on the embeddings to predict the harmfulness of the language.
        from sklearn.linear_model import LogisticRegression
        
        X_train, y_train = train_data.numpy(), train_labels.numpy()
        X_val, y_val = validation_data.numpy(), validation_labels.numpy()
        X_test, y_test = test_data.numpy(), test_labels.numpy()
        
        logistic_model = LogisticRegression(max_iter=5000).fit(X_train, y_train)
        
        if args.log_to_wandb:
            wandb_logging = init_wandb_and_log(args)
            wandb_logging.log({"train_accuracy": logistic_model.score(X_train, y_train), "validation_accuracy": logistic_model.score(X_val, y_val), "test_accuracy": logistic_model.score(X_test, y_test)})
        else:
            print(f"Train Accuracy of the real model: {logistic_model.score(X_train, y_train)}")
            print(f"Validation Accuracy of the real model: {logistic_model.score(X_val, y_val)}")
            print(f"Test Accuracy of the real model: {logistic_model.score(X_test, y_test)}")
        
        ## import mlp from sklearn and train it on the embeddings to predict the harmfulness of the language.
        # from sklearn.neural_network import MLPClassifier
        # mlp_model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=400, verbose=True).fit(X_train, y_train)
        # print(f"Train Accuracy of the real model: {mlp_model.score(X_train, y_train)}")
        # print(f"Validation Accuracy of the real model: {mlp_model.score(X_val, y_val)}")
        # print(f"Test Accuracy of the real model: {mlp_model.score(X_test, y_test)}")
        exit()

    ## now the data loaders are prepared and stuff, now let's train a simple MLP classifier on these embeddings.
    # num_neurons = [2048, 1024, 512, 256, 128]
    # num_neurons = [1024, 512, 256, 128]
    # num_neurons = [512, 256, 128]
    num_neurons = [512, 256]
    project_name = f"harmful_ness_classifier_{args.embeddings_dataset}"
    run_name = f"classifier_lr_{args.learning_rate}_bsz_{args.batch_size}_neurons_{'_'.join([str(x) for x in num_neurons])}"
    train_languages_used = list(train_embeddings.keys())
    ## if there is caesar{some_numer} in the name, replace it with "cr{some_number}", if there is ascii, remane it to asc, if there is hexadecimal, rename it to hex, if there is base64, rename it to b64. If htere is alphanumeric, rename it to alp. If there is "vowel" renmae that to "vow"
    train_languages_used = [x.replace("caesarneg", "crg").replace("caesar", "cr").replace("ascii", "asc").replace("hexadecimal", "hex").replace("base64", "b64").replace("alphanumeric", "alp").replace("vowel", "vow") for x in train_languages_used]
    train_mlp_classifier(args, train_loader, train_languages_used, data_loaders_all_validation_languages_separately, data_loaders_all_test_languages_separately, project_name=project_name, run_name=run_name, only_print_best_accuracy=False, num_neurons=num_neurons, all_external_test_data_loaders=data_loaders_all_external_test_languages_separately)


def load_embeddings_for_testing(args, load_this_test_set):
    test_embeddings = {}
    print(f"\033[93mLoading {load_this_test_set} embeddings for testing...\033[0m")

    if load_this_test_set == 'polyguardmix_test_prompts' or load_this_test_set == 'polyguardmix_test_responses':
        ## now load the test embeddings
        if load_this_test_set == 'polyguardmix_test_prompts':
            model_embeddings_toxic_test = torch.load(f"{model_representation_directory}/polyguardmix_test_harmful_prompt_{language}_representations_{args.llm_model}_{language}.pt", weights_only=True)
            model_embeddings_benign_test = torch.load(f"{model_representation_directory}/polyguardmix_test_benign_prompt_{language}_representations_{args.llm_model}_{language}.pt", weights_only=True)
        elif load_this_test_set == 'polyguardmix_test_responses':
            model_embeddings_toxic_test = torch.load(f"{model_representation_directory}/polyguardmix_test_harmful_response_{language}_representations_{args.llm_model}_{language}.pt", weights_only=True)
            model_embeddings_benign_test = torch.load(f"{model_representation_directory}/polyguardmix_test_benign_response_{language}_representations_{args.llm_model}_{language}.pt", weights_only=True)
        
        model_embeddings_toxic_test = model_embeddings_toxic_test[:, args.hidden_states_representations_layer, :].clone()
        model_embeddings_benign_test = model_embeddings_benign_test[:, args.hidden_states_representations_layer, :].clone()
        
        merged_embeddings_test = torch.cat([model_embeddings_toxic_test, model_embeddings_benign_test], dim=0)
        merged_labels_test = torch.cat([torch.ones(model_embeddings_toxic_test.shape[0], 1, dtype=torch.int), torch.zeros(model_embeddings_benign_test.shape[0], 1, dtype=torch.int)], dim=0)
        assert merged_embeddings_test.shape[0] == merged_labels_test.shape[0]
        test_embeddings[language] = {}
        test_embeddings[language]['toxic'] = model_embeddings_toxic_test
        test_embeddings[language]['benign'] = model_embeddings_benign_test
        print(f"loaded polyguardmix test embeddings for {args.load_separate_test_set_name} using hidden layer {args.hidden_states_representations_layer}")

    elif load_this_test_set == 'multijail_dataset':
        for test_lang in ['en', 'bn', 'ar', 'it', 'jv', 'ko', 'sw', 'th', 'vi', 'zh']:
            model_embeddings_toxic_test = torch.load(f"multijail_dataset/model_representations_multilingual_jailbreaks/multijail_dataset_{test_lang}_representations_{args.llm_model}_{test_lang}.pt", weights_only=True)
            model_embeddings_toxic_test = model_embeddings_toxic_test[:, args.hidden_states_representations_layer, :].clone()
            test_embeddings[load_this_test_set + "_" + test_lang] = {}
            test_embeddings[load_this_test_set + "_" + test_lang]['toxic'] = model_embeddings_toxic_test
            print(f"loaded multijail test embeddings for {load_this_test_set + '_' + test_lang} using hidden layer {args.hidden_states_representations_layer}")

    elif load_this_test_set == 'csrt_dataset':
        model_embeddings_toxic_test = torch.load(f"CSRT_dataset/model_representations_multilingual_jailbreaks/csrt_dataset_representations_{args.llm_model}_code_switched.pt", weights_only=True)
        model_embeddings_toxic_test = model_embeddings_toxic_test[:, args.hidden_states_representations_layer, :].clone()
        test_embeddings[load_this_test_set + "_" + 'csrt'] = {}
        test_embeddings[load_this_test_set + "_" + 'csrt']['toxic'] = model_embeddings_toxic_test
        print("loaded csrt test embeddings")

    elif load_this_test_set == 'xsafety_dataset':
        for test_lang in ['ar', 'bn', 'de', 'en', 'fr', 'hi', 'ja', 'ru', 'sp', 'zh']:
            model_embeddings_toxic_test = torch.load(f"xsafety_dataset/model_representations_multilingual_jailbreaks/xsafety_dataset_representations_{args.llm_model}_{test_lang}.pt", weights_only=True)
            model_embeddings_toxic_test = model_embeddings_toxic_test[:, args.hidden_states_representations_layer, :].clone()
            test_embeddings[load_this_test_set + "_" + test_lang] = {}
            test_embeddings[load_this_test_set + "_" + test_lang]['toxic'] = model_embeddings_toxic_test
            print(f"loaded xsafety test embeddings for {load_this_test_set + '_' + test_lang}")

    elif load_this_test_set == 'rtplx':
        for test_lang in ['en', 'others']:
            model_embeddings_toxic_test = torch.load(f"rtp_lx_dataset/model_representations_multilingual_jailbreaks/rtp_lx_dataset_representations_{args.llm_model}_{test_lang}.pt", weights_only=True)
            model_embeddings_toxic_test = model_embeddings_toxic_test[:, args.hidden_states_representations_layer, :].clone()
            test_embeddings[load_this_test_set + "_" + test_lang] = {}
            test_embeddings[load_this_test_set + "_" + test_lang]['toxic'] = model_embeddings_toxic_test
            print(f"loaded rtplx test embeddings for {load_this_test_set + '_' + test_lang}")

    elif load_this_test_set == 'llm_lat_dataset':
        ## here we will load the entire dataset as test and measure accuracy on that.
        model_representations_directory = "llm_lat_dataset/model_representations_multilingual_jailbreaks/layer_wise_representations"
        directory_name = "llm_lat_dataset"
        for language in ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh']:
            model_embeddings_toxic = torch.load(f'{model_representations_directory}/{directory_name}_harmful_{language}_layer_{args.hidden_states_representations_layer}_representations_{args.llm_model}_{language}.pt', weights_only=True)
            model_embeddings_benign = torch.load(f'{model_representations_directory}/{directory_name}_benign_{language}_layer_{args.hidden_states_representations_layer}_representations_{args.llm_model}_{language}.pt', weights_only=True)
            test_embeddings[language] = {}
            test_embeddings[language]['toxic'] = model_embeddings_toxic
            test_embeddings[language]['benign'] = model_embeddings_benign
            print(f"loaded llm_lat test embeddings for {language}, layer {args.hidden_states_representations_layer}: {model_embeddings_toxic.shape}, {model_embeddings_benign.shape}")

    elif load_this_test_set in ['code_attack_cipher', 'code_attack_python_list', 'code_attack_python_stack', 'code_attack_python_string', 'code_attack_python_list_benign', 'code_attack_python_stack_benign', 'code_attack_python_string_benign']:
        # CodeAttack/prompts/model_representations_multilingual_jailbreaks
        for language in ['code']:
            model_embeddings_toxic = torch.empty((0, args.embedding_dim), dtype=torch.float32).cuda()
            model_embeddings_benign = torch.empty((0, args.embedding_dim), dtype=torch.float32).cuda()
            if "benign" in load_this_test_set:
                model_embeddings_benign = torch.load(f'CodeAttack/benign_prompts/model_representations_multilingual_jailbreaks/{load_this_test_set}_{language}_representations_{args.llm_model}_{language}.pt', weights_only=True)
                model_embeddings_benign = model_embeddings_benign[:, args.hidden_states_representations_layer, :].clone()
                test_embeddings[load_this_test_set + "_" + language] = {}
                test_embeddings[load_this_test_set + "_" + language]['benign'] = model_embeddings_benign
            else:
                model_embeddings_toxic = torch.load(f'CodeAttack/harmful_prompts/model_representations_multilingual_jailbreaks/{load_this_test_set}_{language}_representations_{args.llm_model}_{language}.pt', weights_only=True)
                model_embeddings_toxic = model_embeddings_toxic[:, args.hidden_states_representations_layer, :].clone()
                test_embeddings[load_this_test_set + "_" + language] = {}
                test_embeddings[load_this_test_set + "_" + language]['toxic'] = model_embeddings_toxic
            print(f"loaded {load_this_test_set} test embeddings for {load_this_test_set + '_' + language}, layer {args.hidden_states_representations_layer}: {model_embeddings_benign.shape}, {model_embeddings_toxic.shape}")

    elif load_this_test_set in ['toxigen_dataset', 'xstest_dataset', 'beavertails_rlhf_dataset', 'oai_moderation_dataset', 'jailbreakbench_dataset', 'harmbench_dataset', 'forbidden_questions_dataset', 'toxicchat_dataset', 'simple_safety_tests_dataset', 'saladbench_dataset', 'aya_redteaming_dataset', 'toxicity_jigsaw_dataset', 'toxic_text_dataset', 'advbench_dataset', 'thai_toxicity_tweets', 'ukr_toxicity_dataset']:
        if load_this_test_set == 'aya_redteaming_dataset':
            language_set = ['en', 'fr', 'ru', 'es', 'ar', 'hi', 'sr', 'tl']
        elif load_this_test_set == 'thai_toxicity_tweets':
            language_set = ['th']
        elif load_this_test_set == 'ukr_toxicity_dataset':
            language_set = ['uk']
        else:
            # language_set = ['en', 'es', 'fr', 'de', 'it', 'ar', 'bn', 'jv', 'ko', 'sw', 'th', 'vi', 'zh']
            language_set = ['en', 'fr', 'de', 'es', 'fa', 'ar', 'hr', 'ja', 'pl', 'ru', 'sv', 'th', 'hi', 'it', 'ko', 'bn', 'pt', 'zh', 'he', 'sr', 'da', 'tr', 'el', 'id', 'zu', 'hu', 'eu', 'sw', 'af', 'bs', 'lo', 'ro', 'sl', 'uk', 'fi', 'ms', 'jv', 'cy', 'bg', 'hy', 'is', 'vi', 'si', 'mi', 'gu', 'kn', 'mr', 'ta', 'am', 'te', 'no', 'cs', 'nl', 'caesar1', 'caesar2', 'caesar', 'caesar4', 'caesar5', 'caesar6', 'caesar7', 'caesarneg1', 'caesarneg2', 'caesarneg3', 'caesarneg4', 'caesarneg5', 'caesarneg6', 'caesarneg7', 'ascii', 'hexadecimal', 'base64', 'leet', 'vowel', 'alphanumeric']
            if args.measure_inference_time:
                language_set = ['base64']
        for language in language_set:
            embeddings_toxic_path = f'{load_this_test_set}/model_representations_multilingual_jailbreaks/{load_this_test_set}_harmful_{language}_representations_{args.llm_model}_{language}.pt'
            embeddings_benign_path = f'{load_this_test_set}/model_representations_multilingual_jailbreaks/{load_this_test_set}_benign_{language}_representations_{args.llm_model}_{language}.pt'
            if not os.path.exists(embeddings_toxic_path) and not os.path.exists(embeddings_benign_path):
                print(f"Could not find the embeddings for {load_this_test_set} for {language}. Skipping...")
                continue
            test_embeddings[load_this_test_set + "_" + language] = {}
            if os.path.exists(embeddings_toxic_path):
                model_embeddings_toxic = torch.load(embeddings_toxic_path, weights_only=True)[:, args.hidden_states_representations_layer, :].clone()
                test_embeddings[load_this_test_set + "_" + language]['toxic'] = model_embeddings_toxic
            if os.path.exists(embeddings_benign_path):
                model_embeddings_benign = torch.load(embeddings_benign_path, weights_only=True)[:, args.hidden_states_representations_layer, :].clone()
                test_embeddings[load_this_test_set + "_" + language]['benign'] = model_embeddings_benign
            print(f"loaded {load_this_test_set} test embeddings for {load_this_test_set + '_' + language}, layer {args.hidden_states_representations_layer}")

    else:
        raise NotImplementedError

    return test_embeddings


def evaluate_classifier_harmfulness(args):
    assert args.custom_train_languages is not None      ## we have only implemented it for this case
    assert args.load_separate_test_set_name is not None
    from itertools import permutations
    
    train_languages_used = list(args.custom_train_languages)
    train_languages_used = [x.replace("caesarneg", "crg").replace("caesar", "cr").replace("ascii", "asc").replace("hexadecimal", "hex").replace("base64", "b64").replace("alphanumeric", "alp").replace("vowel", "vow") for x in train_languages_used]
    print(train_languages_used)
    num_neurons = [512, 256]
    model_save_path = None
    lang_permutations = permutations(train_languages_used)
    # if args.embeddings_dataset == 
    for lang_perm in lang_permutations:
        lang_string = '_'.join(lang_perm)
        # model_save_path = f"best_harmfulness_classifiers/{args.llm_model}_{args.embeddings_dataset}_lyr_{args.hidden_states_representations_layer}_tr_{'_'.join(train_languages_used)}_lr_{args.learning_rate}_bsz_{args.batch_size}.pt"
        potential_path = f"best_harmfulness_classifiers/{args.llm_model}_{args.embeddings_dataset}_lyr_{args.hidden_states_representations_layer}_tr_{lang_string}_lr_{args.learning_rate}_bsz_{args.batch_size}.pt"
        # potential_path = f"best_harmfulness_classifiers/llama3.3-70b-instruct_llm_lat_dataset_lyr_57_tr_ar_bn_bs_cs_da_de_en_es_fi_fr_he_hi_hr_hu_id_it_ja_jv_ko_nl_no_pl_pt_ru_sr_sv_sw_th_tr_uk_vi_zh_lr_0.0005_bsz_262144.pt"
        potential_path = f"best_harmfulness_classifiers/llama3.3-70b-instruct_aegis_safety_dataset_lyr_57_tr_en_vi_sw_fr_zh_es_de_th_sv_hi_id_cs_ta_ja_fi_pt_nl_bs_bn_ko_ru_ar_pl_el_mi_it_tr_te_lr_0.0001_bsz_32768.pt"
        # Check if a file exists at this path
        if os.path.exists(potential_path):
            model_save_path = potential_path
            print(f"Found model file: {model_save_path}")
            break # Exit the loop once the file is found
    if model_save_path:
        checkpoint = torch.load(model_save_path)
    else:
        print("Model checkpoint not found. Exiting.")
        return
    loaded_model = create_classifier_harmfulness(args, num_neurons)
    loaded_model.load_state_dict(checkpoint['classifier_state_dict'])
    loaded_model.eval()
    loaded_model = loaded_model.cuda()
    print(f"Loaded the model from {model_save_path}")
    ## count the total number of parameters in the model
    print(f"Total number of parameters in the model: {sum(p.numel() for p in loaded_model.parameters())}")
    
    ## evaluate this on the test set.
    test_embeddings = load_embeddings_for_testing(args)
    _, _, data_loaders_all_test_languages_separately, _ = prepare_data_loaders_for_harmfulness(args, test_embeddings, each_language_separate=True)
    
    ## we now have data loaders for each language in the test set. Pass through the model and get the labels, and get the accuracy
    def eval_model_on_data(data_loaders_all_languages_separately):
        with torch.no_grad():
            accuracy_this_language = {}
            # import ipdb; ipdb.set_trace()
            for language in data_loaders_all_languages_separately:
                predictions = []
                true_labels = []
                for _, (data_here, labels_here) in enumerate(data_loaders_all_languages_separately[language]):
                    data_here, labels_here = data_here.cuda(), labels_here.cuda()
                    output = loaded_model(data_here)
                    predictions.append(torch.sigmoid(output) > 0.5)
                    true_labels.append(labels_here)
                predictions = torch.cat(predictions, dim=0)
                true_labels = torch.cat(true_labels, dim=0)
                accuracy = (predictions == true_labels).float().mean().item()
                accuracy_this_language[language] = round(accuracy * 100, 1)
        return accuracy_this_language
    
    test_accuracy = eval_model_on_data(data_loaders_all_test_languages_separately)
    print(test_accuracy)


def prepare_data_loaders_for_language(args, this_embeddings, train_set=False, trained_eraser=None, extra_embeddings=None, do_regression_for_lease=False):
    ## this_embeddings is a dictionary with keys as languages and values as the embeddings of that language. Make a dataloader for the entire dataset, with the language as the label.
    global all_languages, cipher_languages, natural_languages
    all_data = []
    all_harmfulness_labels = []
    for language in this_embeddings:
        all_data.append(this_embeddings[language])
        all_harmfulness_labels.append(torch.tensor([all_languages.index(language)] * this_embeddings[language].shape[0]))

    all_data = torch.cat(all_data, dim=0)
    all_harmfulness_labels = torch.cat(all_harmfulness_labels, dim=0)
    
    # import ipdb; ipdb.set_trace()
    if args.modify_embeddings_using_LEACE and train_set:        ## we will only modify the embeddings of the training set, and send the eraser for the test set.
        from concept_erasure import LeaceEraser
        # all_data = compute_changed_X_for_LEACE(all_data, all_harmfulness_labels)      ## YOUR IMPLEMENTATION
        print("Modifying the embeddings using LEACE 1")
        
        if do_regression_for_lease:
            assert len(all_languages) == all_harmfulness_labels.max().item() + 1
            X = all_data
            Y = all_harmfulness_labels
            ## create the test embeddings also
            if extra_embeddings is not None:
                extra_data = []
                extra_labels = []
                for language in extra_embeddings:
                    extra_data.append(extra_embeddings[language])
                    extra_labels.append(torch.tensor([all_languages.index(language)] * extra_embeddings[language].shape[0]))
                
                extra_data = torch.cat(extra_data, dim=0)
                extra_labels = torch.cat(extra_labels, dim=0)
                X_extra = extra_data
                Y_extra = extra_labels
            
            from sklearn.linear_model import LogisticRegression
            # import ipdb; ipdb.set_trace()
            real_lr = LogisticRegression(max_iter=400).fit(X, Y)
            beta = torch.from_numpy(real_lr.coef_)
            print("Fitted a real logistic regression model!", beta.max(), beta.min())
            print(f"Train Accuracy of the real model: {real_lr.score(X, Y)}")
            if extra_embeddings is not None:
                print(f"Test Accuracy of the real model: {real_lr.score(X_extra, Y_extra)}")
            
            print("Modifying the embeddings using LEACE 2")
            eraser = LeaceEraser.fit(X, Y)
            X_ = eraser(X)
            
            print("Modified the embeddings using LEACE!")
            null_lr = LogisticRegression(max_iter=400, tol=0.0).fit(X_.numpy(), Y)
            beta_null = torch.from_numpy(null_lr.coef_)
            print("Fitted a null logistic regression model!", beta_null.max(), beta_null.min())
            print(f"Accuracy of the null model: {null_lr.score(X_.numpy(), Y)}")
            if extra_embeddings is not None:
                print(f"Test Accuracy of the null model: {null_lr.score(eraser(X_extra).numpy(), Y_extra)}")
            exit()
        
        eraser_train = LeaceEraser.fit(all_data.cuda(), torch.nn.functional.one_hot(all_harmfulness_labels, num_classes=all_harmfulness_labels.max().item() + 1).cuda())
        all_data_new = eraser_train(all_data.cuda()).cpu()
        print("Modified the embeddings using LEACE for the train set ")
        this_embeddings_dataset = TensorDataset(all_data_new, all_harmfulness_labels)
        data_loader = DataLoader(this_embeddings_dataset, batch_size=8192*8, shuffle=True, num_workers=4)
        return data_loader, eraser_train
    
    elif args.modify_embeddings_using_LEACE and not train_set:
        assert trained_eraser is not None
        all_data_new = trained_eraser(all_data.cuda()).cpu()
        print("Modified the embeddings using LEACE using already trained eraser ")
        this_embeddings_dataset = TensorDataset(all_data_new, all_harmfulness_labels)
        data_loader = DataLoader(this_embeddings_dataset, batch_size=8192*8, shuffle=True, num_workers=4)
        return data_loader
    
    else:
        this_embeddings_dataset = TensorDataset(all_data, all_harmfulness_labels)
        data_loader = DataLoader(this_embeddings_dataset, batch_size=8192*8, shuffle=True, num_workers=4)
        return data_loader


def train_models_to_predict_language(args, train_loader, validation_loader, test_loader):
    ## now the data loaders are prepared and stuff, now let's train a simple logistic regression classifier on these embeddings.
    if args.train_classifier_language_logistic:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        X_train, y_train = train_loader.dataset.tensors[0].cpu().numpy(), train_loader.dataset.tensors[1].cpu().numpy()
        X_val, y_val = validation_loader.dataset.tensors[0].cpu().numpy(), validation_loader.dataset.tensors[1].cpu().numpy()
        X_test, y_test = test_loader.dataset.tensors[0].cpu().numpy(), test_loader.dataset.tensors[1].cpu().numpy()
        print("Training a logistic regression classifier on the embeddings to predict the language of the embeddings.")
        classifier_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        train_accuracy = accuracy_score(y_train, classifier_model.predict(X_train)) * 100.0
        validation_accuracy = accuracy_score(y_val, classifier_model.predict(X_val)) * 100.0
        test_accuracy = accuracy_score(y_test, classifier_model.predict(X_test)) * 100.0
        print(f"Train Accuracy of the LR model: {train_accuracy:.2f}, Validation Accuracy of the LR model: {validation_accuracy:.2f}, Test Accuracy of the LR model: {test_accuracy:.2f}")
        
        if args.log_to_wandb:
            os.environ["WANDB_API_KEY"] = "65b10491413acd011c96d46acd3990854fded569"
            import wandb
            project_name = 'language-detection-accuracy-LR'
            wandb.init(project=project_name,
                name = f"logistic_classifier_{args.embeddings_dataset}_seed_{args.random_seed}_layer_{args.hidden_states_representations_layer}",
                config={
                        "embeddings_dataset": args.embeddings_dataset,
                        "random_seed": args.random_seed,
                        "hidden_states_representations_layer": args.hidden_states_representations_layer,
                    }
                )
            wandb.log({"train_accuracy": train_accuracy, "validation_accuracy": validation_accuracy, "test_accuracy": test_accuracy, "hidden_states_representations_layer": args.hidden_states_representations_layer})

        # classifier_model = LogisticRegression(max_iter=10, warm_start=True)  # Train one iteration per epoch
        # for epoch in range(50):
        #     classifier_model.fit(X_train, y_train)  # Train one epoch (thanks to warm_start)

        #     # Compute accuracies
        #     train_accuracy = accuracy_score(y_train, classifier_model.predict(X_train))
        #     validation_accuracy = accuracy_score(y_val, classifier_model.predict(X_val))
        #     test_accuracy = accuracy_score(y_test, classifier_model.predict(X_test))

        #     print(f"\033[93mEpoch {epoch*10 + 1}: Train Acc = {train_accuracy*100:.2f}%, Val Acc = {validation_accuracy*100:.2f}%, Test Acc = {test_accuracy*100:.2f}%\033[0m")
    
    elif args.train_classifier_language_mlp:
        classifier_model = nn.Sequential(
            nn.Linear(args.embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(all_languages))
        )
        
        # initialize the weights of the classifier model with xavier initialization
        for layer in classifier_model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        criterion_classifier = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(classifier_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
        
        classifier_model = classifier_model.cuda()
        print("Model training starts now!")
        ## TRAINING THE CLASSIFIER. AT THE END OF EVERY 10 EPOCHS, MEASURE THE VALIDATION AND TEST ACCURACIES.
        for epoch in range(args.num_epochs):
            classifier_model.train()
            for batch_idx, (data_train, labels_train) in enumerate(train_loader):
                data, labels = data_train.cuda(), labels_train.cuda()
                optimizer.zero_grad()
                output = classifier_model(data)
                loss = criterion_classifier(output, labels)
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0 or epoch == args.num_epochs - 1:
                def eval_model_on_data(data_loader):
                    classifier_model.eval()
                    with torch.no_grad():
                        predictions = []
                        true_labels = []
                        for batch_idx, (data_here, labels_here) in enumerate(data_loader):
                            data_here, labels_here = data_here.cuda(), labels_here.cuda()
                            output = classifier_model(data_here)
                            predictions.append(torch.argmax(output, dim=1))
                            true_labels.append(labels_here)
                        predictions = torch.cat(predictions, dim=0)
                        true_labels = torch.cat(true_labels, dim=0)
                        accuracy = (predictions == true_labels).float().mean().item()
                    return round(accuracy * 100, 1)
                
                validation_accuracy = eval_model_on_data(validation_loader)
                test_accuracy = eval_model_on_data(test_loader)
                
                print(f"Epoch: {epoch}, Validation Accuracy: {validation_accuracy}, Test Accuracy: {test_accuracy}")
                scheduler.step(validation_accuracy)


class TransformModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        if isinstance(args.use_larger_transform_model, int):
            num_layers = args.use_larger_transform_model
            layers = []
            
            for _ in range(num_layers):
                layers.append(nn.Linear(embed_dim, embed_dim))
                layers.append(nn.ReLU())
            
            layers.append(nn.Linear(embed_dim, embed_dim))
            self.net = nn.Sequential(*layers)
            
            ## initialize the weights of the classifier model with xavier initialization
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
            self.args = args
        else:
            self.net = nn.Parameter(torch.eye(embed_dim))
    
    def forward(self, x):
        if isinstance(args.use_larger_transform_model, int):
            return self.net(x)
        else:
            return x @ self.net


class ClassifierModel(nn.Module):
    
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )   
        ## initialize the weights of the classifier model with xavier initialization
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.classifier(x)
    

def get_model_id(args):
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
    elif args.llm_model == "qwen-2.5-32B-instruct":
        model_id = "Qwen/Qwen2.5-32B-Instruct"
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
    return model_id


def load_tokenizer_and_model(args, load_from_saved_directory=None, use_deepspeed=False, modifier_model=None):
    model_id = get_model_id(args)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "llama3-70b" in args.llm_model or "llama3.1-70b" in args.llm_model or "llama3.3-70b" in args.llm_model or "qwen-2.5-72B-instruct" in args.llm_model:
        num_layers = 80
    elif "qwen2.5-32b" in args.llm_model:
        num_layers = 64
    elif "llama3-8b" in args.llm_model or "llama3.1-8b" in args.llm_model:
        num_layers = 32
    elif 'mistral-nemo-12b' in args.llm_model:
        num_layers = 40
    elif 'multilingual-e5-large' in args.llm_model:
        num_layers = 24      ## this is a contrastive model, so num layers don't matter 
    elif 'qwen2.5-32B-instruct' in args.llm_model:
        num_layers = 64
    else:
        raise NotImplementedError
    
    if load_from_saved_directory is not None:
        model = AutoModelForCausalLM.from_pretrained(load_from_saved_directory, device_map='auto', torch_dtype=torch.bfloat16) 
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return tokenizer, model, device, num_layers


def load_dataset_and_embeddings(args, only_dataset=False):
    ## Let's just pick some random sentences from flores200 datasets, we already have the last token embeddings for these sentences. 
    if len(args.other_language_translate_to_english) == 1:
        other_language = args.other_language_translate_to_english[0]
    
    if args.embeddings_dataset == "flores200_dataset":
        dataset_file = "flores200_dataset/benign_en.txt"
        # other_language = "en"
        other_language_dataset_file = f"flores200_dataset/benign_{other_language}.txt"
        loaded_dataset = open(dataset_file, "r").readlines()        ## use indices: 35, 53, 93, 137, 181, 248, 350, 400, 461, 504
        other_language_loaded_dataset = open(other_language_dataset_file, "r").readlines()
        # embeddings_file = f"flores200_dataset/model_representations/flores200_dataset_benign_representations_{args.llm_model}_en.pt"
        other_language_embeddings_file = f"flores200_dataset/model_representations/flores200_dataset_benign_representations_{args.llm_model}_{other_language}.pt"
    
    elif args.embeddings_dataset == "wildguard" or args.embeddings_dataset == "wildguard_filtered_for_length_and_category":
        dataset_file = "wildguard_datasets/mWildGuardMix-train-tower-nllb-v2.1/wildguard-train-English.parquet"     ## this is the target language which is english. 
        loaded_dataset = pd.read_parquet(dataset_file)
        ## drop any rows where prompt is null
        loaded_dataset = loaded_dataset.dropna(subset=['prompt'])
        assert loaded_dataset['id'].is_monotonic_increasing # and other_language_loaded_dataset['id'].is_monotonic_increasing and len(loaded_dataset) == len(other_language_loaded_dataset)
        loaded_dataset = loaded_dataset['prompt'].tolist()
        
        other_langs_embedding_files = {}
        for other_language in sorted(args.other_language_translate_to_english):
            # if other_language in natural_languages:
            #     other_language_dataset_file = f"wildguard_datasets/mWildGuardMix-train-tower-nllb-v2.1/wildguard-train-{languages_names[other_language]}.parquet"
            # else:
            #     other_language_dataset_file = f"wildguard_datasets/mWildGuardMix-train-tower-nllb-v2.1/wildguard-train-{other_language}.parquet"
            other_language_embeddings_file = f"wildguard_datasets/mWildGuardMix-train-tower-nllb-v2.1/model_last_token_representations_multilingual_jailbreaks/wildguard_dataset_{other_language}_last_token_representations_{args.llm_model}_{other_language}.pt"
            other_langs_embedding_files[other_language] = other_language_embeddings_file
            print(f"Loading {other_language} dataset from {other_language_embeddings_file}")
        
        # other_language_loaded_dataset = pd.read_parquet(other_language_dataset_file)
        # other_language_loaded_dataset = other_language_loaded_dataset.dropna(subset=['prompt'])
        # other_language_loaded_dataset = other_language_loaded_dataset['prompt'].tolist()
        # print(f"Loaded {len(loaded_dataset)} sentences from {dataset_file} for {other_language}")   # and {len(other_language_loaded_dataset)} sentences from {other_language_dataset_file}")
        # embeddings_file = f"wildguard_datasets/mWildGuardMix-train-tower-nllb-v2.1/model_representations_multilingual_jailbreaks/wildguard_dataset_en_representations_{args.llm_model}_en.pt"

    else:
        raise NotImplementedError(f"Dataset {args.embeddings_dataset} not supported")
    
    if only_dataset:
        return loaded_dataset
    
    selected_indices = list(range(len(loaded_dataset)))
    last_layer_embeds_other_langs = {}
    for other_language in sorted(args.other_language_translate_to_english):
        other_language_selected_embeddings = torch.load(other_langs_embedding_files[other_language], weights_only=True)
        assert other_language_selected_embeddings.shape == (len(selected_indices), 81, 8192) if args.llm_model == "llama3.3-70b-instruct" else (len(selected_indices), 32, 4096) if args.llm_model == "llama3.1-8b-instruct" else (len(selected_indices), 64, 5120)
        # if args.llm_model == "llama3.3-70b-instruct":
        #     best_layer = 58
        # elif args.llm_model == "llama3.1-8b-instruct":
        #     best_layer = 32     # 14
        last_layer_embeds = other_language_selected_embeddings[:, args.hidden_states_representations_layer, :].clone().float()
        last_layer_embeds_other_langs[other_language] = last_layer_embeds
        assert last_layer_embeds.shape[0] == len(selected_indices) == len(loaded_dataset)
    
    return loaded_dataset, last_layer_embeds_other_langs


def build_modifier_model(args, hidden_size):
    layers = []
    for _ in range(args.modifier_model_num_layers):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_size, hidden_size))
    if args.modifier_model_num_layers == 0:
        layers.append(nn.ReLU())
    modifier = nn.Sequential(*layers)
    return modifier


def make_dataset_splits(args, num_in_context, len_loaded_dataset, device):
    percentage_train_examples = 0.6
    percentage_val_examples = 0.2
    percentage_test_examples = 0.2
    ## of the remaining dataset, take these percentages and create train, validation and test sets
    num_train_examples = int(percentage_train_examples * len_loaded_dataset)
    num_val_examples = int(percentage_val_examples * len_loaded_dataset)
    num_test_examples = int(percentage_test_examples * len_loaded_dataset)
    print(f"Using {num_in_context} in-context examples, Num train examples: {num_train_examples}, Num val examples: {num_val_examples}, Num test examples: {num_test_examples}")
    
    all_idxs = list(range(len_loaded_dataset))
    train_idxs = list(set(random.sample(all_idxs, num_train_examples)))
    rem = [i for i in all_idxs if i not in train_idxs]
    val_idxs   = list(set(random.sample(rem, num_val_examples)))
    test_idxs  = list(set(random.sample([i for i in rem if i not in val_idxs], num_test_examples)))
    
    idx_shape_tensors = torch.tensor([len(train_idxs), len(val_idxs), len(test_idxs)], device=device, dtype=torch.long)
    split_tensors = [torch.tensor(x, dtype=torch.long).to(device, non_blocking=True) for x in (train_idxs, val_idxs, test_idxs)]
    return train_idxs, val_idxs, test_idxs, idx_shape_tensors, split_tensors


def get_model_name_from_args(args):
    langs_concat = "_".join(args.other_language_translate_to_english)
    if langs_concat == 'caesar_caesar1_ascii_hexadecimal_base64_leet_vowel_alphanumeric_caesar2_caesar4_caesar5_caesar6_caesarneg1_caesarneg2_caesarneg3_caesarneg4_caesarneg5_caesarneg6_atbash_vigenere_keyboard_reverse':
        ## then take the first 3 characters of each language
        langs_concat = "_".join([lang[:3] for lang in args.other_language_translate_to_english])
        model_name = "all_languages_model"
    elif langs_concat == "caesar6_caesar2_atbash_caesarneg2_leet_caesar4_vigenere_caesarneg6_caesarneg5_caesar5_caesar":
        model_name = "first_half_languages_model"
    elif langs_concat == "hexadecimal_caesarneg4_keyboard_caesar1_base64_alphanumeric_vowel_caesarneg3_caesarneg1_ascii_reverse":
        model_name = "second_half_languages_model"
    # elif langs_concat in cipher_languages:
    #     ## this is just one language
    #     model_name = "one_language_model"
    elif langs_concat == "vigenere_caesar1_caesar2_caesar4_caesar5_caesar6_caesarneg1_caesarneg2_caesarneg3_caesarneg4_caesarneg5":
        model_name = "good_performance_languages"
    elif langs_concat == "vigenere_caesar1_caesar2_caesar_caesar4_caesar5_caesar6_caesarneg1_caesarneg2_caesarneg3_caesarneg4_caesarneg5_caesarneg6":
        model_name = "vig_and_all_caesar_languages"
    elif langs_concat == "vigenere_caesar1_caesar2_caesar4_caesar5_caesar6_caesarneg1_caesarneg2_caesarneg3_caesarneg4_caesarneg5_atbash":
        model_name = "good_performance_languages_with_atbash"
    elif langs_concat == "vigenere_caesar1_caesar2_caesar4_caesar5_caesar6_caesarneg1_caesarneg2_caesarneg3_caesarneg4_caesarneg5_atbash_alphanumeric_ascii":
        model_name = "good_performance_languages_with_atbash_alphanumeric_ascii"
    elif langs_concat == "vigenere_caesar1_caesar2_caesar_caesar4_caesar5_caesar6_caesarneg1_caesarneg2_caesarneg3_caesarneg4_caesarneg5_caesarneg6_atbash_alphanumeric_ascii_leet_vowel":
        model_name = "all_caesar_atbash_alphanumeric_ascii_leet_vowel_vigenere"
    elif langs_concat == "vigenere_caesar1_caesar2_caesar_caesar4_caesar5_caesar6_caesarneg1_caesarneg2_caesarneg3_caesarneg4_caesarneg5_caesarneg6_atbash_alphanumeric_ascii_leet_vowel_hexadecimal_keyboard":
        langs_concat = "_".join([lang[:3] for lang in args.other_language_translate_to_english])
        model_name = "all_except_reverse_and_base64"
    elif langs_concat == "caesar1_caesar_caesar5_caesarneg1_caesarneg3_caesarneg5":
        model_name = "odd_caesar_languages"
        args.other_language_translate_to_english = ["caesar1", "caesar2", "caesar", "caesar4", "caesar5", "caesar6", "caesarneg1", "caesarneg2", "caesarneg3", "caesarneg4", "caesarneg5", "caesarneg6"]
    elif langs_concat == "caesar2_caesar4_caesar6_caesarneg2_caesarneg4_caesarneg6":
        model_name = "even_caesar_languages"
        args.other_language_translate_to_english = ["caesar1", "caesar2", "caesar", "caesar4", "caesar5", "caesar6", "caesarneg1", "caesarneg2", "caesarneg3", "caesarneg4", "caesarneg5", "caesarneg6"]
    elif langs_concat == "caesar1_caesar_caesar5_caesar7_caesar9_caesarneg1_caesarneg3_caesarneg5_caesarneg7_caesarneg9_base64_leet_morse":
        model_name = "odd_caesar_languages_base64_leet_morse"
        args.other_language_translate_to_english = cipher_languages
    elif langs_concat == "caesar1_caesar_caesar5_caesar7_caesar9_caesarneg1_caesarneg3_caesarneg5_caesarneg7_caesarneg9_base64_vowel_atbash":
        model_name = "odd_caesar_languages_base64_vowel_atbash"
        args.other_language_translate_to_english = cipher_languages
    elif langs_concat == "caesar1_caesar_caesar5_caesar7_caesar9_caesarneg1_caesarneg3_caesarneg5_caesarneg7_caesarneg9_base64_hexadecimal_vowel":
        model_name = "odd_caesar_languages_base64_hexadecimal_vowel"
        args.other_language_translate_to_english = cipher_languages
    elif langs_concat == "caesar1_caesar_caesar5_caesar7_caesar9_caesarneg1_caesarneg3_caesarneg5_caesarneg7_caesarneg9_hexadecimal_vowel_leet":
        model_name = "odd_caesar_languages_base64_hexadecimal_vowel_leet"
        args.other_language_translate_to_english = cipher_languages
    elif langs_concat == "caesar1_caesar_caesar5_caesar7_caesar9_caesarneg1_caesarneg3_caesarneg5_caesarneg7_caesarneg9_atbash_vowel_leet":
        model_name = "odd_caesar_languages_base64_atbash_vowel_leet"
        args.other_language_translate_to_english = cipher_languages
    elif langs_concat == "caesar1_caesar_caesar5_caesar7_caesar9_caesarneg1_caesarneg3_caesarneg5_caesarneg7_caesarneg9_alphanumeric_leet_morse":
        model_name = "odd_caesar_languages_alphanumeric_leet_morse"
        args.other_language_translate_to_english = cipher_languages
    elif langs_concat == "caesar1_caesar_caesar5_caesar7_caesar9_caesarneg1_caesarneg3_caesarneg5_caesarneg7_caesarneg9_ascii_leet_morse":
        model_name = "odd_caesar_languages_ascii_leet_morse"
        args.other_language_translate_to_english = cipher_languages
    elif langs_concat == "caesar1_caesar_caesar5_caesar7_caesar9_caesarneg1_caesarneg3_caesarneg5_caesarneg7_caesarneg9_base64_leet_ascii":
        model_name = "odd_caesar_languages_base64_leet_ascii"
        args.other_language_translate_to_english = cipher_languages
    elif langs_concat == "caesar1_caesar_caesar5_caesar7_caesar9_caesarneg1_caesarneg3_caesarneg5_caesarneg7_caesarneg9_hexadecimal_leet_morse":
        model_name = "odd_caesar_languages_hexadecimal_leet_morse"
        args.other_language_translate_to_english = cipher_languages
    elif langs_concat == "caesar1_caesar_caesar5_caesar7_caesar9_caesarneg1_caesarneg3_caesarneg5_caesarneg7_caesarneg9_hexadecimal_leet_base64":
        model_name = "odd_caesar_languages_hexadecimal_leet_base64"
        args.other_language_translate_to_english = cipher_languages
    elif langs_concat == "caesar1_caesar_caesar5_caesar7_caesar9_caesarneg1_caesarneg3_caesarneg5_caesarneg7_caesarneg9_atbash_leet_base64":
        model_name = "odd_caesar_languages_atbash_leet_base64"
        args.other_language_translate_to_english = cipher_languages
    elif langs_concat == "caesar1_caesar_caesar5_caesar7_caesar9_caesarneg1_caesarneg3_caesarneg5_caesarneg7_caesarneg9_hexadecimal_leet_atbash":
        model_name = "odd_caesar_languages_hexadecimal_leet_atbash"
        args.other_language_translate_to_english = cipher_languages
    elif langs_concat == "caesar1_caesar5_caesar7_caesar9_caesarneg1_caesarneg5_caesarneg7_caesarneg9_ascii_base64_leet_vowel_alphanumeric_atbash":
        model_name = "some_odd_caesar_languages_ascii_base64_leet_vowel_alphanumeric_atbash"
        args.other_language_translate_to_english = cipher_languages
    elif langs_concat == "caesar1_caesar5_caesar9_caesarneg1_caesarneg5_caesarneg7_ascii_hexadecimal_base64_leet_vowel_alphanumeric_atbash_morse":
        model_name = "some_odd_caesar_languages_ascii_hexadecimal_base64_leet_vowel_alphanumeric_atbash_morse"
        args.other_language_translate_to_english = cipher_languages
    elif langs_concat == "ascii_alphanumeric":
        model_name = "ascii_alphanumeric"
        args.other_language_translate_to_english = cipher_languages
    elif langs_concat == "morse":
        model_name = "just_morse"
        args.other_language_translate_to_english = cipher_languages
    elif langs_concat == "base64_hexadecimal":
        model_name = "base64_hexadecimal"
        args.other_language_translate_to_english = cipher_languages
    elif langs_concat == "vowel_leet":
        model_name = "vowel_leet"
        args.other_language_translate_to_english = cipher_languages
    else:
        print(langs_concat)
        raise NotImplementedError
    return model_name, langs_concat


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_languages_randomly', action='store_true')
    parser.add_argument('--split_languages_randomly_test', action='store_true')
    parser.add_argument('--split_only_cipher_languages_randomly', action='store_true')
    parser.add_argument('--split_datapoints_for_english_only', action='store_true')
    parser.add_argument('--split_datapoints_for_these_languages_only', action='store_true')
    parser.add_argument('--split_train_test_along_only_languages', action='store_true')
    parser.add_argument('--split_train_test_along_only_datapoints', action='store_true')
    parser.add_argument('--train_test_already_split_load_separately', action='store_true')
    parser.add_argument('--load_separate_test_set_name', nargs='+', default=None)
    parser.add_argument('--split_train_test_along_datapoints_and_use_separate_test_set', action='store_true')
    
    parser.add_argument('--split_custom_languages_for_train_test', action='store_true')
    parser.add_argument('--custom_train_languages', type=str, nargs='+', default=None)
    parser.add_argument('--custom_validation_languages', type=str, nargs='+', default=None)
    parser.add_argument('--custom_test_languages', type=str, nargs='+', default=None)
    parser.add_argument('--split_train_test_along_languages_and_datapoints', action='store_true')
    parser.add_argument('--split_train_test_along_languages_and_datapoints_60_train', action='store_true')
    parser.add_argument('--split_train_test_along_languages_and_datapoints_50_train', action='store_true')
    parser.add_argument('--split_train_test_along_languages_and_datapoints_40_train', action='store_true')
    parser.add_argument('--split_train_test_along_languages_and_datapoints_30_train', action='store_true')
    parser.add_argument('--split_train_test_along_languages_and_datapoints_20_train', action='store_true')
    parser.add_argument('--split_train_test_along_languages_and_datapoints_10_train', action='store_true')
    parser.add_argument('--split_train_test_along_languages_and_datapoints_5_train', action='store_true')
    parser.add_argument('--split_train_test_along_languages_and_datapoints_2_train', action='store_true')
    parser.add_argument('--split_train_test_along_languages_and_datapoints_1_train', action='store_true')

    parser.add_argument('--split_train_test_along_languages_and_datapoints_50_datapoints', action='store_true', help='instead of taking fixed percentage of datapoints, take fixed number of datapoints: 50 datapoints')
    parser.add_argument('--split_train_test_along_languages_and_datapoints_100_datapoints', action='store_true', help='instead of taking fixed percentage of datapoints, take fixed number of datapoints: 100 datapoints')

    parser.add_argument('--random_seed_for_language_split', type=int, default=0)
    parser.add_argument('--random_seed_for_datapoint_split', type=int, default=0)

    parser.add_argument('--embeddings_dataset', type=str, default=None)
    parser.add_argument('--hidden_states_representations_layer', type=int, default=None)
    parser.add_argument('--use_last_token_embeddings', action='store_true', default=False)
    parser.add_argument('--pass_through_modifier_model', action='store_true', default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--log_epochs', type=int, default=10)

    ## This one only trains the harmfulness classifier
    parser.add_argument('--train_classifier_harmfulness', action='store_true')
    parser.add_argument('--save_best_harmfulness_model', action='store_true')
    parser.add_argument('--evaluate_classifier_harmfulness', action='store_true')
    parser.add_argument('--train_classifier_harmfulness_logistic', action='store_true')

    ## This one trains the language prediction classifier
    parser.add_argument('--train_classifier_language_logistic', action='store_true')
    parser.add_argument('--train_classifier_language_mlp', action='store_true')
    parser.add_argument('--modify_embeddings_using_LEACE', action='store_true')
    parser.add_argument('--modify_embeddings_using_min_crosscov', action='store_true')


    ## This one trains the harmfulness classifier while minimizing the cross-covariance between the transformed embeddings and the language labels.
    parser.add_argument('--train_harmfulness_min_crosscov', action='store_true')
    parser.add_argument('--train_harmfulness_leace_transform', action='store_true')
    parser.add_argument('--language_loss_weight', type=float, default=1e-4)
    parser.add_argument('--add_no_transform', action='store_true')
    parser.add_argument('--use_second_implementation_of_cross_cov', action='store_true')
    parser.add_argument('--use_larger_transform_model', type=int, default=None, choices=[0, 1, 2, 3, 4, 5, 6], help="0: one layer, and as it becomes larger, we are using larger and larger models.")
    parser.add_argument('--bring_embeddings_closer_to_english', action='store_true')
    parser.add_argument('--test_bring_embeddings_closer_to_english', action='store_true')
    parser.add_argument('--another_test_bring_embeddings_closer_to_english', action='store_true')
    
    parser.add_argument('--compute_CS_LRD_embeddings_classifier', action='store_true')
    parser.add_argument('--lrd_rank', type=int, default=10)

    parser.add_argument('--train_adversarial_language_adaption', action='store_true', help='train an adversarial model that learns to adapt the embeddings of a language X to language agnostic')
    parser.add_argument('--use_hidden_states_representations', action='store_true', help='if true, use the hidden states representations')
    parser.add_argument('--use_heads_concatenated_representations', action='store_true', help='if true, use the concatenated representations instead of the hidden states. We found them to be better for distinguishing between paried and unpaired languages')
    parser.add_argument('--heads_concatenated_representation_layer', type=int, help='if true, use the concatenated representations of the heads at layer 39 or 41', default=39, choices=[39, 41])
    parser.add_argument('--use_heads_separate_representations', action='store_true', help='if true, use the separate representations of the heads')
    parser.add_argument('--heads_separate_top_k_heads', type=int, help='if true, use the top k heads for each language', default=64)
    parser.add_argument('--get_test_performance_of_trained_models', action='store_true', help='if true, get the test performance of the trained models on the test languages')
    parser.add_argument('--llm_model', type=str, choices=["llama3.3-70b-instruct-old", "llama3.1-8b-instruct", "llama3.3-70b-instruct", "qwen2.5-32b-instruct"], help='choose the model to use for generating responses', default=None)
    parser.add_argument('--DA_feature_extractor_num_hidden_layers', type=int, default=2)
    parser.add_argument('--DA_feature_extractor_hidden_dim1', type=int, default=1024)
    parser.add_argument('--DA_feature_extractor_hidden_dim2', type=int, default=512)
    parser.add_argument('--DA_feature_extractor_hidden_dim3', type=int, default=256)
    parser.add_argument('--DA_feature_extractor_hidden_dim4', type=int, default=256)
    parser.add_argument('--DA_feature_extractor_hidden_dim5', type=int, default=256)
    parser.add_argument('--DA_feature_extractor_hidden_dim6', type=int, default=256)
    parser.add_argument('--DA_classifier_num_hidden_layers', type=int, default=1)
    parser.add_argument('--DA_classifier_hidden_dim1', type=int, default=256)
    parser.add_argument('--DA_classifier_hidden_dim2', type=int, default=128)
    parser.add_argument('--DA_classifier_hidden_dim3', type=int, default=128)
    parser.add_argument('--DA_domain_classifier_num_hidden_layers', type=int, default=1)
    parser.add_argument('--DA_domain_classifier_hidden_dim1', type=int, default=256)
    parser.add_argument('--DA_domain_classifier_hidden_dim2', type=int, default=128)
    parser.add_argument('--lambda_max_domain_adaptation', type=float, default=1e-4)
    
    parser.add_argument('--modifier_model_num_layers', type=int, help='Number of layers in the modified model', default=None)
    parser.add_argument('--modifier_model_train_epochs', type=int, help='Number of epochs to train the modified model', default=None)
    parser.add_argument('--remove_translation_prompt', action='store_true', help='Remove the translation prompt from the input')
    parser.add_argument('--get_metrics_from_wandb', action='store_true')
    parser.add_argument('--log_to_wandb', action='store_true')
    
    
    parser.add_argument('--finetune_entire_llm_for_translating_to_english', action='store_true', help='Finetune the entire LLM to translate from another language to en')
    parser.add_argument('--use_finetuned_model_to_produce_last_token_embedding', action='store_true')
    parser.add_argument('--train_model_using_last_token_embedding_translated_to_english', action='store_true')
    parser.add_argument('--train_model_using_last_token_embedding_translated_to_english_all_languages_loaded_mixed', action='store_true')
    parser.add_argument('--use_best_checkpoint', action='store_true')
    parser.add_argument('--use_last_checkpoint', action='store_true')
    parser.add_argument('--other_language_translate_to_english', type=str, nargs='+', default=None)
    
    parser.add_argument("--deepspeed", metavar="CFG", default=None, help="Path to DeepSpeed JSON; omit to run without DS")
    parser.add_argument("--measure_inference_time", action="store_true", help="Measure inference time")
    parser.add_argument("--finetune_on_external_datasets", action="store_true", help="Finetune on external datasets")
    parser.add_argument("--external_datasets_for_finetuning", type=str, nargs='+', default=None, help="External datasets for finetuning")
    parser.add_argument("--num_few_shot_examples", type=int, default=0, help="Number of few-shot examples to use for finetuning")
    
    ## for Vision-Language Dataset

    args = parser.parse_args()
    
    print("Random Seed for language: ", args.random_seed_for_language_split, "Random Seed for datapoint: ", args.random_seed_for_datapoint_split)
    
    if args.llm_model in ["llama3.3-70b-instruct-old", "llama3.3-70b-instruct"]:
        args.embedding_dim = 8192
        args.llm_model = "llama3.3-70b-instruct"
        if args.hidden_states_representations_layer is None:
            args.hidden_states_representations_layer = 58
    elif args.llm_model in ["llama3.1-8b-instruct"]:
        args.embedding_dim = 4096
        if args.hidden_states_representations_layer is None:
            args.hidden_states_representations_layer = 14
    elif args.llm_model in ["qwen2.5-32b-instruct"]:
        if args.hidden_states_representations_layer is None:
            args.hidden_states_representations_layer = 32
        args.embedding_dim = 5120
        
    print(f"Using {args.llm_model} at hidden states representations layer {args.hidden_states_representations_layer}")
    
    if args.train_classifier_harmfulness or args.train_classifier_harmfulness_logistic or args.compute_CS_LRD_embeddings_classifier or args.train_adversarial_language_adaption or args.evaluate_classifier_harmfulness:
        assert sum([args.split_train_test_along_only_languages, args.split_train_test_along_languages_and_datapoints, args.split_train_test_along_languages_and_datapoints_1_train, args.split_train_test_along_languages_and_datapoints_2_train, args.split_train_test_along_languages_and_datapoints_5_train, args.split_train_test_along_languages_and_datapoints_10_train, args.split_train_test_along_languages_and_datapoints_40_train, args.split_train_test_along_languages_and_datapoints_20_train, args.split_train_test_along_languages_and_datapoints_30_train, args.split_train_test_along_languages_and_datapoints_50_train, args.split_train_test_along_languages_and_datapoints_60_train, args.split_train_test_along_languages_and_datapoints_50_datapoints, args.split_train_test_along_languages_and_datapoints_100_datapoints, args.split_datapoints_for_english_only, args.split_train_test_along_only_datapoints, args.train_test_already_split_load_separately, args.split_train_test_along_datapoints_and_use_separate_test_set]) == 1
        
        if args.compute_CS_LRD_embeddings_classifier:
            compute_CS_LRD_embeddings_classifier(args)
        elif args.train_adversarial_language_adaption:
            train_adversarial_language_adaption(args)
        elif args.train_classifier_harmfulness:
            train_vanilla_classifier_on_embeddings_to_predict_harmfulness(args)
        elif args.evaluate_classifier_harmfulness:
            evaluate_classifier_harmfulness(args)
        else: raise NotImplementedError

    elif args.train_classifier_language_logistic or args.train_classifier_language_mlp:
        train_logistic_regression_on_embeddings_to_predict_language(args)
    
    elif args.train_harmfulness_min_crosscov or args.train_harmfulness_leace_transform:
        train_models_on_embeddings_to_predict_harmfulness_while_min_cross_covariance_or_leace_transform(args)
    
    elif args.get_metrics_from_wandb:
        get_metrics_from_wandb(args)
    
    elif args.test_bring_embeddings_closer_to_english:
        test_bring_embeddings_closer_to_english(args)
        
    elif args.another_test_bring_embeddings_closer_to_english:
        another_test_bring_embeddings_closer_to_english(args)
    
    elif args.bring_embeddings_closer_to_english:
        assert args.other_language_translate_to_english is not None
        bring_embeddings_closer_to_english(args)
    
    elif args.finetune_entire_llm_for_translating_to_english:
        assert args.other_language_translate_to_english is not None
        finetune_entire_llm_for_translating_to_english(args)
    
    elif args.use_finetuned_model_to_produce_last_token_embedding:
        assert args.other_language_translate_to_english is not None
        use_finetuned_model_to_produce_last_token_embedding(args)
        
    elif args.train_model_using_last_token_embedding_translated_to_english:
        assert args.other_language_translate_to_english is not None
        assert sum([args.use_best_checkpoint, args.use_last_checkpoint]) == 1
        train_model_using_last_token_embedding_translated_to_english_individually(args)
    
    elif args.train_model_using_last_token_embedding_translated_to_english_all_languages_loaded_mixed:
        assert args.other_language_translate_to_english is not None
        assert sum([args.use_best_checkpoint, args.use_last_checkpoint]) == 1
        train_model_using_last_token_embedding_translated_to_english_all_languages_loaded_mixed(args)
    
    else:
        raise NotImplementedError
    

if __name__ == "__main__":
    main()
