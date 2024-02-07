import faiss
import numpy as np
import torch
import torch.nn as nn
import math
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from transformers import default_data_collator
import torch.nn.functional as F
import networkx as nx
import faiss
from sklearn.neighbors import kneighbors_graph
import pandas as pd
from datasets import Dataset
import math


def extract_emb2(model, tokenizer, batch, device):
    pad_token_id = tokenizer.pad_token_id
    with torch.no_grad():
        outputs = model(
            input_ids = batch['input_ids'].to(device),
            attention_mask = batch['attention_mask'].to(device),
            token_type_ids = batch['token_type_ids'].to(device)
        )

        x = outputs.last_hidden_state

        pad_mask = (batch['input_ids'] != pad_token_id).float()
        pad_mask = pad_mask.unsqueeze(-1).expand(x.size()).float().to(device)

        x = x * pad_mask

        mean_pooling = x[:,1:-1,:].sum(dim=1) / pad_mask[:,1:-1].sum(dim=1)
    

    return mean_pooling


def get_tree_weights(dataset, args):
    layer_weights = None

    if args.dataset == 'SetFit/20_newsgroups':
        neumeric_str = dict(zip(dataset['label'], dataset['label_text']))
        keys = list(neumeric_str.keys())
        keys.sort()
        ordered_numeric_str = {i: neumeric_str[i] for i in keys}
        labels_str = list(ordered_numeric_str.values())
        layers = torch.Tensor([len(lb.split('.')) for lb in labels_str])
        layer_weights = abs(layers.unsqueeze(-1) - layers.unsqueeze(-2))

        dicts = {"alt": "alternative",
                "atheism": "atheism",
                "comp": "computer",
                "graphics": "graphics",
                "rec": "recreation",
                "sport": "sport",
                "hockey": "hockey",
                "sci": "science",
                "crypt": "cryptography",
                "electronics": "electronics",
                "med": "medcine",
                "space": "space",
                "soc": "sociology",
                "religion": "religion",
                "christian": "christian",
                "politics": "politics",
                "guns": "guns",
                "mideast": "mideast",
                "misc": "misc",
                "ms-windows": "ms-windows",
                "ibm": "ibm",
                "pc": "pc",
                "hardware": "hardware",
                "mac": "mac",
                "windows": "windows",
                "x": "x",
                "forsale": "forsale",
                "autos": "autos",
                "motorcycles": "motorcycles",
                "sport": "sport",
                "baseball": "baseball",
                "os": "operating system",
                "sys": "system",
                "talk": "talk"
                }

        new_labels_str = [",".join(list(map(dicts.get, w))) for i in labels_str for w in [i.split('.')]]
        

    return layers, new_labels_str


def under_sample(dataset, args, var):
    df = pd.DataFrame(dataset)
    newdf = pd.DataFrame()

    if args.dataset == 'DeveloperOats/DBPedia_Classes':
        newdf = pd.DataFrame()
        l1, l2 = dataset['l1'], dataset['l2']
        l1_l2 = [l1[i]+','+l2[i] for i in range(len(l1))]
        df['l1_l2'] = l1_l2
        label_cnt = dict(df[var].value_counts())
        keys, values = list(label_cnt.keys()), list(label_cnt.values())
        for i in range(len(label_cnt)):
            subdf = df[df[var]==keys[i]].reset_index(drop=True)
            if values[i] > 7000:
                subcnt = math.floor(values[i] * 0.1)
                newdf = pd.concat([newdf, subdf.iloc[:subcnt,:]]).reset_index(drop=True)
            elif 4000 < values[i] <= 7000:
                subcnt = math.floor(values[i] * 0.5)
                newdf = pd.concat([newdf, subdf.iloc[:subcnt,:]]).reset_index(drop=True)
            else:
                newdf = pd.concat([newdf, subdf]).reset_index(drop=True)
    
    elif args.dataset == 'go_emotions':
        label_cnt = dict(df[var].value_counts())
        keys, values = list(label_cnt.keys()), list(label_cnt.values())
        min_val = min(values)

        for i in range(len(label_cnt)):
            subdf = df[df[var]==keys[i]].reset_index(drop=True)
            newdf = pd.concat([newdf, subdf.sample(n=min_val, random_state=42)]).reset_index(drop=True)

    return Dataset.from_pandas(newdf)


def description():
    dicts = {
        "recreation,sport,hockey": "In the latest recreation and sport news, hockey enthusiasts are buzzing with excitement as teams gear up for an intense season filled with thrilling matches and adrenaline-pumping action on the ice.",
        "alternative,atheism": "In the realm of alternative perspectives, the latest atheism news explores the growing movement of individuals who embrace a secular worldview, challenging traditional beliefs and fostering intellectual discourse on matters of religion and spirituality.",
        "computer,graphics": "Breaking news in the world of computer graphics reveals groundbreaking advancements in technology, unleashing stunning visual experiences that push the boundaries of realism and immerse users in captivating digital realms like never before.",
        "science,cryptography": "Exciting developments in the field of science and cryptography are making headlines, as innovative breakthroughs in encryption techniques promise enhanced security measures to protect sensitive information in an increasingly interconnected and data-driven world.",
        "science,electronics": "The world of science and electronics is abuzz with the latest news, showcasing cutting-edge discoveries and advancements in electronic components, paving the way for smaller, faster, and more efficient devices that are revolutionizing various industries and transforming the way we interact with technology.",
        "science,medcine": "Exciting breakthroughs in the realm of science and medicine are making headlines, as researchers unveil groundbreaking treatments, innovative technologies, and novel therapies that hold the potential to revolutionize healthcare, improve patient outcomes, and shape the future of medical practice.",
        "science,space": "Captivating discoveries in the realm of science and space are making headlines, as astronomers unveil breathtaking celestial phenomena, unravel the mysteries of the universe, and delve deeper into the exploration of distant galaxies, igniting our curiosity and expanding our understanding of the cosmos.",
        "sociology,religion,christian": "In the realm of sociology and religion, the latest news focuses on the dynamic interactions between Christian faith and societal dynamics, shedding light on how believers navigate contemporary challenges, forge meaningful connections, and shape the fabric of communities while adhering to their religious beliefs and values.",
        "talk,politics,guns": "The latest news on the talk surrounding politics and guns highlights the ongoing debates and discussions surrounding firearm regulations, Second Amendment rights, and public safety, fueling a heated discourse on the role of guns in society and the pursuit of effective policies to address these complex issues.",
        "talk,politics,mideast": "The talk surrounding politics in the Middle East has intensified as key stakeholders engage in negotiations, diplomatic efforts, and strategic discussions aimed at fostering stability, resolving conflicts, and shaping the future of the region amidst intricate geopolitical dynamics.",
        "talk,politics,misc": "The latest news on the talk surrounding politics delves into a myriad of diverse and compelling topics, ranging from policy debates, electoral campaigns, and international relations to social movements, governance challenges, and emerging trends, fueling a vibrant exchange of ideas and shaping the political landscape in fascinating and unpredictable ways.",
        "talk,religion,misc": "The current news on the talk surrounding religion encompasses a wide range of captivating discussions, exploring diverse faith traditions, interfaith dialogue, ethical debates, religious freedoms, and the role of religion in contemporary society, fostering an enriching exchange of perspectives and contributing to our understanding of spirituality in an ever-changing world.",
        "computer,operating system,ms-windows,misc": "The latest news in the realm of computer operating systems shines a spotlight on MS Windows, as the tech giant unveils exciting updates, enhanced features, and improved functionality, aiming to provide users with a seamless and intuitive computing experience while addressing emerging challenges and staying at the forefront of technological innovation.",
        "computer,system,ibm,pc,hardware": "In the realm of computer systems and hardware, the latest news centers around IBM's groundbreaking advancements, unveiling cutting-edge technologies and next-generation components for PCs that promise enhanced performance, improved efficiency, and a transformative computing experience, propelling the industry into new frontiers of innovation.",
        "computer,system,mac,hardware": "Exciting developments in the world of computer systems and hardware bring the latest news on Apple's Mac lineup, showcasing powerful and sleek devices equipped with state-of-the-art components, remarkable performance capabilities, and innovative features, solidifying Mac's reputation as a leading choice for professionals and enthusiasts seeking unparalleled computing experiences.",
        "computer,windows,x": "The latest news in the realm of computer technology shines a spotlight on Windows X, a highly anticipated operating system release from Microsoft, promising a fresh and intuitive user interface, advanced features, and seamless integration across devices, offering users a transformative computing experience tailored to their evolving needs.",
        "misc,forsale": "In the world of miscellaneous items for sale, exciting news emerges as a diverse array of products and services hit the market, ranging from unique collectibles, cutting-edge gadgets, and exclusive deals to one-of-a-kind experiences, providing consumers with a wealth of options to explore and fulfill their desires.",
        "recreation,autos": "In the realm of recreation and autos, the latest news showcases thrilling developments in the world of vehicles, from high-performance sports cars and eco-friendly electric models to innovative off-road adventures and luxury travel experiences, captivating automotive enthusiasts and igniting a passion for unforgettable journeys on the open road.",
        "recreation,motorcycles": "Revving up excitement in the realm of recreation, the latest news in motorcycles unveils thrilling advancements, from sleek designs and cutting-edge technology to exhilarating riding experiences, showcasing a dynamic landscape that caters to motorcycle enthusiasts and fuels their passion for speed, adventure, and the open road.",
        "recreation,sport,baseball": "In the realm of recreation and sports, baseball fans are eagerly following the latest news as teams gear up for an action-packed season filled with intense competition, incredible athleticism, and unforgettable moments that capture the spirit of America's favorite pastime."
    }
    return dicts