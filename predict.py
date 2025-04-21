import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from ModelArchitecture.DUCK_Net import create_model


def load_and_preprocess_image(image_path, img_height, img_width):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((img_height, img_width))
    image = np.array(image) / 255.0  # 归一化
    return np.expand_dims(image, axis=0)  # 增加批次维度
def predict_single_image(model, image_path, img_height, img_width):
    processed_image = load_and_preprocess_image(image_path, img_height, img_width)
    prediction = model.predict(processed_image)
    return prediction[0]
def predict_images_in_directory(model, images_dir, img_height, img_width, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = [os.path.join(images_dir, fname) for fname in os.listdir(images_dir) if
                   fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_path in tqdm(image_paths, desc="Processing images"):
        prediction = predict_single_image(model, image_path, img_height, img_width)

        prediction_mask = (prediction[..., 0] * 255).astype(np.uint8)
        prediction_mask = cv2.resize(prediction_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

        output_filename = os.path.basename(image_path)#.replace('.png', '_mask.png')
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, prediction_mask)
        print(f"Saved prediction to {output_path}")
def load_trained_model(model_path, img_height, img_width, input_channels, output_classes, starting_filters):
    model = create_model(img_height, img_width, input_channels, output_classes, starting_filters)
    model.load_weights(model_path)
    # model = tf.keras.models.load_model(model_path, custom_objects={'dice_metric_loss': dice_metric_loss})
    return model

if __name__ == '__main__':
    # 配置参数
    img_shape = [256, 256]
    input_channels = 3
    output_classes = 1
    starting_filters = 8
    # model_path = "logs/2025_04_20_20_21_03/best_epoch_model.h5"
    # images_dir = "data/BUSI-256/images/"
    # output_dir = "data/BUSI-256/predict/"
    model_path = "logs/2025_04_20_22_44_19/best_epoch_model.h5"
    images_dir='data/isic2018/test/images/'
    output_dir='data/isic2018/test/predict/'
    model = load_trained_model(model_path, img_shape[0], img_shape[1], input_channels, output_classes, starting_filters)
    predict_images_in_directory(model, images_dir, img_shape[0], img_shape[1], output_dir)
