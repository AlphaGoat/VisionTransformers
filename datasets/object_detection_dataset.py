import os
import torch
from PIL import Image
from pycocotools.coco import COCO


class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        Args:
            annotations_file (str): Path to the annotations file.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.annotations = self.load_annotations(annotations_file)

    def load_annotations(self, annotations_file):
        coco = COCO(annotations_file)
        annotations = []
        for img_id in coco.imgs.keys():
            img_info = coco.imgs[img_id]
            ann_ids = coco.getAnnIds(img_id)
            anns = coco.loadAnns(ann_ids)
            boxes = []
            for i in range(len(anns)):
                xmin = anns[i]['bbox'][0]
                ymin = anns[i]['bbox'][1]
                xmax = anns[i]['bbox'][0] + anns[i]['bbox'][2]
                ymax = anns[i]['bbox'][1] + anns[i]['bbox'][3]
                boxes.append([xmin, ymin, xmax, ymax])
            class_ids = [ann['category_id'] for ann in anns]
            annotations.append({
                'image_id': img_info['file_name'],
                'boxes': boxes,
                'class_id': class_ids
            })
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.img_dir, ann['image_id'])
        image = self.load_image(img_path)
        boxes = torch.tensor(ann['boxes'], dtype=torch.float32)
        classes = torch.tensor(ann['class_id'], dtype=torch.int64)

        sample = {'image': image, 'boxes': boxes, 'class_id': classes, 'image_id': ann['image_id']}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, img_path):
        # Placeholder for image loading logic
        # This should return a tensor representation of the image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # HWC to CHW