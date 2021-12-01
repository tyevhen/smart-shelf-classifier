import torch
import classification
import json

# SAMPLE PAYLOAD {"url": "https://res.cloudinary.com/dudfwhpoz/image/upload/v1637952627/temp_name/ux6bjnunhhfwgk5t5xif.jpg"}

class PythonPredictor:

    def __init__(self, config):
        """ Download pretrained model. """
        self.model = classification.load_model('resnet50_14_20210607_2148.pt')

    def classify_images(self, images):
        """
        :param images: (tensor): image tensor of shape (nof_images, channels, w, h)
        :return: preds, probs
        """
        images = classification.images_to_tensor([classification.transform_reshape_image(x) for x in images])
        with torch.no_grad():
            self.model.eval()
            out = self.model(images)
            ps = torch.exp(out)
            top_preds, top_classes = ps.topk(1, dim=1)
            preds = [
                self.model.idx_to_class[class_] for class_ in top_classes.cpu().numpy().flatten()
            ]
            probs = top_preds.cpu().numpy().flatten()
            return preds, probs

    def predict(self, payload):
        """ Run a model based on url input. """
        print(payload)
        img = classification.load_image(payload["url"])
        preds, probs = self.classify_images([img])

        return json.dumps({"preds": str(preds), "probs": str(probs)})

