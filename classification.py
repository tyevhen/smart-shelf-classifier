import mlflow
import torch
from torchvision import models, transforms
from torchvision.transforms.functional import crop
from PIL import Image
# from memory_profiler import profile


def classify_images(images):
    if isinstance(images,  list):
        images = images_to_tensor([transform_reshape_image(x) for x in images])
    else:
        images = transform_reshape_image(images)
    return classify_image(images)


def classify_image(image):
    """
    :param image: (tensor): image tensor of shape (nof_images, channels, w, h)
    :return: preds, probs
    """
    with torch.no_grad():
        model.eval()
        out = model(image)
        ps = torch.exp(out)
        top_preds, top_classes = ps.topk(1, dim=1)
        preds = [
            model.idx_to_class[class_] for class_ in top_classes.cpu().numpy().flatten()
        ]
        probs = top_preds.cpu().numpy().flatten()
        return preds, probs


def images_to_tensor(image_list):
    return torch.cat(image_list, 0)


def get_transform(mode='test'):
    transforms_obj = {
        # Train uses data augmentation
        'train':
            transforms.Compose([
                transforms.Resize(256),
                transforms.Lambda(center_top_crop),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
                transforms.ToTensor(),

            ]),
        # Validation does not use augmentation
        'valid':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.Lambda(center_top_crop),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),

        # Test does not use augmentation
        'test':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.Lambda(center_top_crop),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
    }
    return transforms_obj.get(mode)


def center_top_crop(img):
    new_left = (img.size[0] - 224) / 2
    return crop(img, 0, new_left, 224, 224)


def freeze_weights(model):
    for param in model.parameters():
        param.requires_grad = False


# @profile
def load_model(path):
    global model
    checkpoint = torch.load(path)
    model = models.resnet50(pretrained=False)
    freeze_weights(model)
    model.fc = checkpoint['fc']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']
    
    

def transform_reshape_image(img):
    img_tensor = get_transform()(img)
    return img_tensor.view(1, 3, 224, 224)


def load_image(url):
    # TODO implement load_image_by_url()
    # image_raw = load_img_by_url()
    img_raw = Image.open(url)
    return img_raw

# images = ['sample_images/dagelsvviw.jpg', 'sample_images/yqggboetfh.jpg']
# raw_images = [load_image(img) for img in images]
# preds, probs = classify_images(raw_images, model_tuned)

if __name__ == '__main__':
    load_model('resnet50_14_20210607_2148.pt')
    # with mlflow.start_run() as run: 
    mlflow.set_tracking_uri("http://localhost:5000")
    # mlflow.pytorch.save_model(model, 'classifier')
    mlflow.pytorch.log_model(model, "model")
        
        
