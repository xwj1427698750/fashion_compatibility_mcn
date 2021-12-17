import json
import os
import random
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class CategoryDataset(Dataset):
    """Dataset for polyvore with 4 categories(upper, bottom, shoe, bag),
    each suit has at least 3 items, the missing items will use a mean image.

    Args:
        root_dir: Directory stores source images
        data_file: A json file stores each outfit index and description
        data_dir: Directory stores data_file and mean_images
        transform: Operations to transform original images to fix size
        use_mean_img: Whether to use mean images to fill the blank part
        neg_samples: Whether generate negative sampled outfits
    """
    def __init__(self, root_dir="../data/images2/", data_file='train.json', data_dir="../data", transform=None,
                 use_mean_img=True, neg_samples=True):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.transform = transform
        self.use_mean_img = use_mean_img
        self.data = json.load(open(os.path.join(data_dir, data_file)))
        self.data = [(k, v) for k, v in self.data.items()]  # k:套装id,v:{套装对象}
        self.neg_samples = neg_samples  # if True, will randomly generate negative outfit samples
    
        self.vocabulary, self.word_to_idx = [], {}
        self.word_to_idx['UNK'] = len(self.word_to_idx)
        self.vocabulary.append('UNK')
        with open(os.path.join(self.data_dir, 'vocab.dat')) as f:
            for line in f:
                name = line.strip().split()[0]
                if name not in self.word_to_idx:
                    self.word_to_idx[name] = len(self.word_to_idx)
                    self.vocabulary.append(name)

    def __getitem__(self, index):
        """It could return a positive suits or negative suits"""
        set_id, parts = self.data[index]
        if random.randint(0, 1) and self.neg_samples:  # random.randint(0, 1) 随机生成0和1
            to_change = list(parts.keys())  # random choose negative items
        else:
            to_change = []
        imgs = []
        labels = []
        names = []
        # 负样本的生成方式： 套装里的每一件衣服都是重新选择的，然后组成了一件新的套装
        for part in ['upper', 'bottom', 'shoe', 'bag']:
            if part in to_change:  # random choose a image from dataset with same category
                choice = self.data[index]
                while (choice[0] == set_id) or (part not in choice[1].keys()):
                    choice = random.choice(self.data)
                img_path = os.path.join(self.root_dir, str(choice[1][part]['index'])+'.jpg')
                names.append(torch.LongTensor(self.str_to_idx(choice[1][part]['name'])))
                labels.append('{}_{}'.format(choice[0], choice[1][part]['index']))
            elif part in parts.keys():  # 正样本，原有套装的数据
                img_path = os.path.join(self.root_dir, str(parts[part]['index'])+'.jpg')
                names.append(torch.LongTensor(self.str_to_idx(parts[part]['name'])))
                labels.append('{}_{}'.format(set_id, parts[part]['index']))
            else:
                continue
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
        input_images = torch.stack(imgs)  # 沿着第0维度拼接图片 [N,C,H,W], N = len(imgs)
        is_compat = (len(to_change) == 0)
        return input_images, names, set_id, labels, is_compat

    def __len__(self):
        return len(self.data)

    """ 根据文本描述的长度返回对应长度的向量 """
    def str_to_idx(self, name):
        return [self.word_to_idx[w] if w in self.word_to_idx else self.word_to_idx['UNK']
            for w in name.split()]

    def get_fitb_quesiton(self, index, option_len=4):
        """Generate fill in th blank questions.
        Return:
            images: 4 parts of a outfit
            labels: store if this item is empty
            question_part: which part to be changed
            options: 3 other item with the same category,
            expect original composition get highest score
        """
        set_id, parts = self.data[index]  # set_id：套装id, parts:套装的组成部分的描述
        question_part = random.choice(list(parts))  # list(parts)获得parts中的keys形成的列表，然后从中随机选择一个， eg:  question_part：upper
        question_id = "{}_{}".format(set_id, parts[question_part]['index'])  # eg:  parts[question_part]['index'] : be9627e0c2e43ee96017e288b03eed96(图片的编号)
        imgs = []
        labels = []
        for part in ['upper', 'bottom', 'shoe', 'bag']:
            if part in parts.keys():
                img_path = os.path.join(self.root_dir, str(parts[part]['index'])+'.jpg')
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                imgs.append(img)
                labels.append('{}_{}'.format(set_id, parts[part]['index']))
        items = torch.stack(imgs)  # 正确的搭配

        option_ids = [set_id]
        options = []
        option_labels = []
        while len(option_ids) < option_len:
            option = random.choice(self.data)
            if (option[0] in option_ids) or (question_part not in option[1]):
                continue
            else:
                option_ids.append(option[0])
                img_path = os.path.join(self.root_dir, str(option[1][question_part]['index'])+'.jpg')
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                options.append(img)
                option_labels.append("{}_{}".format(option[0], option[1][question_part]['index']))

        # Return 4 options for question, 3 incorrect options
        return items, labels, question_part, question_id, options, option_labels

def collate_fn(data):
    """Need custom a collate_fn"""
    data.sort(key=lambda x: x[0].shape[0], reverse=True)
    images, names, set_ids, labels, is_compat = zip(*data)
    is_compat = torch.LongTensor(is_compat)
    names = sum(names, [])
    images = torch.stack(images)
    return (images, names, set_ids, labels, is_compat)



# Test the loader
if __name__ == "__main__":
    img_size = 224
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor()
    ])
    d = CategoryDataset(transform=transform, use_mean_img=True)
    loader = DataLoader(d, 4, shuffle=True, num_workers=4, collate_fn=collate_fn)
