import os
import albumentations as albu
import numpy as np
from torch.utils.data import Dataset as BaseDataset
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import copy

class Dataset(BaseDataset):

    def __init__(
            self,
            images_dir,
            masks_dir,
            #classes=None,
            model_image_size,
            augmentation=None,
            preprocessing=None,

    ):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.msds = sorted(os.listdir(masks_dir))
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.msds]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.model_image_size = model_image_size

    def __getitem__(self, i):

        # read data
        image = Image.open(self.images_fps[i])
        mask = Image.open(self.masks_fps[i])

        if image.size != self.model_image_size:
           image = image.resize(self.model_image_size)
           mask = mask.resize(self.model_image_size, Image.NEAREST)

        image = np.array(image)
        mask = np.array(mask)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask) #Why do we need to preprocess mask?
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class CustomBatchDataset(Dataset):
    def __init__(self, dataset, batches):
        self.dataset = dataset
        self.batches = batches

    def __getitem__(self, index):
        images, masks = [], []

        # Extract images and masks from the dataset for the given batch indices
        for i in self.batches[index]:
            image, mask = self.dataset[i]
            if isinstance(image, np.ndarray):
                images.append(torch.tensor(image))
            else:
                images.append(image.clone().detach())

            if isinstance(mask, np.ndarray):
                masks.append(torch.tensor(mask))
            else:
                masks.append(mask.clone().detach())

        # Convert lists of images and masks to tensors
        return torch.stack(images), torch.stack(masks)

    def __len__(self):
        return len(self.batches)


class BalancedDataset(BaseDataset):

    def __init__(self, images_dir, masks_dir, model_image_size, n_classes, augmentation=None,
                 preprocessing=None, method_weights='inverse', cut_off_empty = None):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.msds = sorted(os.listdir(masks_dir))
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.msds]
        print('Number of images in dataset: ', len(self.masks_fps))

        self.cut_off_empty = cut_off_empty
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.model_image_size = model_image_size
        self.n_classes = n_classes #number of classes including class "0"

        # Calculate the global frequencies of each class across the dataset
        self.global_frequencies, self.class_indices = self.calculate_global_frequencies_and_make_index()

        self.method_weights = method_weights
        self.global_class_weights = self.calculate_class_weights(frequencies = self.global_frequencies.items(), method=self.method_weights)


    def print_info (self):
        print('Global frequencies: ', self.global_frequencies)
        print('Global weights ( method =', self.method_weights, ')', self.global_class_weights)
        print('')
        print('ORIGINAL DATASET STRUCTURE:')
        for cls, indices in self.class_indices.items():
            print(f"Class {cls} has {len(indices)} dominant class patches.")
        if hasattr(self, 'oversampled_class_weights'):
            print('Global weights ( method =', self.method_weights, ')', self.oversampled_class_weights)
        if hasattr(self, 'class_indices_oversample'):
            print('')
            print('OVERSAMPLED DATASET STRUCTURE:')
            for cls, indices in self.class_indices_oversample.items():
                print(f"Class {cls} has {len(indices)} dominant class patches.")

    def calculate_class_weights(self, frequencies, method='median'):

        print('Start calculation of global weights for classes based on global pixel frequencies...')
        # Exclude the ignore_index (class 0) from calculations
        filtered_frequencies = {cls: freq for cls, freq in frequencies if cls != 0}

        total_pixels = sum(filtered_frequencies.values())
        class_weights = {}
        class_weights[0] = 0

        if method == 'inverse':
            for cls, freq in filtered_frequencies.items():
                class_weights[cls] = 1 / (np.log(1.02 + (freq / total_pixels)))

        elif method == 'median':
            median_freq = np.median(list(filtered_frequencies.values()))
            for cls, freq in filtered_frequencies.items():
                class_weights[cls] = median_freq / freq

        elif method == 'sklearn':
            # Getting the class labels and their frequencies from filtered_frequencies
            classes = list(filtered_frequencies.keys())
            sample_counts = list(filtered_frequencies.values())

            weights = compute_class_weight('balanced', classes=classes, y=np.repeat(classes, sample_counts))
            class_weights.update(dict(zip(classes, weights)))

        # Assigning weight 0 for the ignored class

        return list(class_weights.values())
    def calculate_oversampled_frequencies_and_weights(self, cut_off_empty = None):

        frequencies = {i: 0 for i in range(self.n_classes)}  # n_classes classes including class 0
        print("Starting calculation of oversampled frequencies for class indices..")

        for cls, indices in tqdm(self.class_indices_oversample.items()):
            for idx in indices:
                # Load the corresponding mask image
                mask = Image.open(self.masks_fps[idx])
                mask = mask.resize((100, 100), Image.NEAREST)  # resize to make calculations quicker
                mask = np.array(mask)

                # Count pixels belonging to the current class
                for i in range(self.n_classes):
                    frequencies[i] += np.sum(mask == i)

        self.oversampled_frequencies = frequencies

        self.oversampled_class_weights = self.calculate_class_weights(frequencies=self.oversampled_frequencies.items(),
                                                                 method=self.method_weights)
        print("Oversampled class weights: ")
        print(self.oversampled_class_weights)

    def calculate_global_frequencies_and_make_index(self, cut_off_empty = None):
        class_indices = {i: [] for i in range(1, self.n_classes)}
        frequencies = {i: 0 for i in range(self.n_classes)}  # n_classes classes including class 0
        print("Starting calculation of global frequencies..")
        for idx, mask_fp in tqdm(enumerate(self.masks_fps), total=len(self.masks_fps), desc="Processing masks", unit="mask"):
            mask = Image.open(mask_fp)
            mask = mask.resize((100, 100), Image.NEAREST)  # resize to make calculations quicker
            mask = np.array(mask)

            # Check if the number of non-zero pixels is less than 30% of total pixels
            if self.cut_off_empty:
                non_zero_pixels = np.sum(mask != 0)
                total_pixels = mask.size
                if non_zero_pixels / total_pixels < self.cut_off_empty:
                    continue

            # Count frequencies of each class in the image
            freq = {i: (mask == i).sum() for i in range(self.n_classes)}

            for cls, count in freq.items():
                frequencies[cls] += count

            # Find the dominant class (ignoring class "0") and update class_indices
            dominant_class = max({i: freq_val for i, freq_val in freq.items() if i != 0}, key=freq.get)
            class_indices[dominant_class].append(idx)

        return frequencies, class_indices

    def get_oversampled_batches(self, batch_size, target_file_number_oversample, stop_on_rarest=False, shuffle=False):
        class_indices_working = copy.deepcopy(self.class_indices)

        for cls, indices in class_indices_working.items():
            check_class = False
            # If current class has less than target_file_number_oversample indices
            while len(indices) < target_file_number_oversample:
                # Extend the list
                indices.extend(indices)
                check_class = True

            # Crop the list to the target size
            if check_class == True:
                class_indices_working[cls] = indices[:target_file_number_oversample]

        self.class_indices_oversample = class_indices_working

        self.calculate_oversampled_frequencies_and_weights(cut_off_empty=self.cut_off_empty)

        # Now, construct batches using the oversampled class indices
        return self.construct_balanced_batches(batch_size, 'oversample', stop_on_rarest, shuffle)

    def construct_balanced_batches(self, batch_size, indices = 'original', stop_on_rarest=False, shuffle=False):

        if indices == 'original':
            class_indices_ = copy.deepcopy(self.class_indices)
        elif indices == 'oversample':
            class_indices_ = copy.deepcopy(self.class_indices_oversample)

        #Make oversampling of rare classes.

        # Sort classes by their global frequency
        classes_sorted_by_rarity = sorted(self.global_frequencies, key=self.global_frequencies.get)
        classes_sorted_by_rarity = [cls for cls in classes_sorted_by_rarity if cls != 0]

        # Shuffle based on the option selected
        if shuffle:
            print('Shuffling batches ..')
            for indices in class_indices_.values():
                np.random.shuffle(indices)

        batches = []
        batch = []
        print('Start forming balanced batches ..')
        if stop_on_rarest:
            print('Early stop on exhausting the rarest class is activated ..')
        while any(class_indices_.values()):
            for cls in classes_sorted_by_rarity:
                if not class_indices_[cls]:
                    if stop_on_rarest:
                        return batches
                    else:
                        continue
                batch.append(class_indices_[cls].pop())
                if len(batch) == batch_size:
                    #print(batch)
                    batches.append(batch)
                    batch = []
        # Any leftover images that don't make up a full batch will be discarded.
        return batches

    def __getitem__(self, i):
        image = Image.open(self.images_fps[i])
        mask = Image.open(self.masks_fps[i])

        if image.size != self.model_image_size:
            image = image.resize(self.model_image_size)
            mask = mask.resize(self.model_image_size, Image.NEAREST)

        image = np.array(image)
        mask = np.array(mask)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [
        albu.RandomRotate90(p=0.3),
        albu.HorizontalFlip(p=0.3),  # original: 0.2
        albu.VerticalFlip(p=0.3),
        albu.RandomBrightness(limit=0.25, p=0.5),
        albu.RandomGamma(gamma_limit=(90, 110), p=0.2),  # original: 0.2
        albu.RandomContrast(limit=0.3, p=0.5),
        albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=0, p=0.5),
        #albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=20, interpolation=1,
        #                      border_mode=4, value=None, mask_value=None, always_apply=False,
        #                      approximate=False, same_dxdy=False, p=0.2),
        #albu.Affine(scale=(0.95,1.05), p=0.2),
        # albu.CLAHE(p=1.0),  # original:not useda
        # albu.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, >
        # albu.Emboss (alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=1.0), # original:no>
        albu.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=False, always_apply=False, p=0.3),
        # albu.Blur(blur_limit=2, always_apply=False, p=0.25),
        # original:not used
        albu.JpegCompression(quality_lower=60, quality_upper=90, always_apply=False, p=0.3),  # orig>
        # albu.PixelDropout (dropout_prob=0.2, per_channel=False, drop_value=0, mask_drop_value=None, a>
    ]
    return albu.Compose(train_transform)

def to_tensor_x(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def to_tensor_y(y, **kwargs):
    return torch.Tensor(y).to(torch.int64)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor_x, mask=to_tensor_y),
    ]
    return albu.Compose(_transform)
