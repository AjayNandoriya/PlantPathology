import os
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import json
from sklearn import metrics
import wget
import zipfile

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

def download_dataset():
    zip_file_url = r"http://download2266.mediafire.com/kf0wsokjbbzg/phtv0nr53jy8dkk/plant-pathology-2020-fgvc7.zip"
    filename = os.path.basename(zip_file_url)
    if not filename:
        print("downloading dataset %s", filename)
        filename = wget.download(zip_file_url)
    print("extracting dataset")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(".")



def predict_one(model, image_id):
    """Predict single input
    Args:
        model (keras model): keras model loaded
        image_id (str): image id
    Returns:
        float ndarray (1x4): output prediction
    """
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image_file_name = os.path.join('images', image_id + '.jpg')
    if not os.path.isfile(image_file_name):
        return np.zeros((1, 4))

    image = Image.open(image_file_name)

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    # image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)

    return prediction


def predict_all(model, image_ids):
    """Predict all inputs
    Args:
        model (keras model): Keras model loaded
        image_ids (list): list of string image ids
    Returns:
        predictions (float ndarray Nx4): output predictions
    """
    TOTAL_IDS = len(image_ids)
    predictions = np.ndarray(shape=(TOTAL_IDS, 4), dtype=np.float32)
    for k, image_id in enumerate(image_ids):
        prediction = predict_one(model, image_id)
        predictions[k, :] = prediction[0, :]
    return predictions


def load_input(input_csv_file_name):
    """Load Input csv file
    Args:
        input_csv_file_name (str): input csv file name
    Returns:
        inputs (list): list of string of image file ids
        outputs (float ndarray Nx4): gt data values for all category
    """
    TOTAL_CATEGORIES = 4
    inputs = []
    outputs = []
    with open(input_csv_file_name) as fid:
        header = fid.readline()
        for line in fid.readlines():
            row = line.strip().split(',')
            image_id = row[0]
            inputs.append(image_id)
            if len(row) == TOTAL_CATEGORIES + 1:
                gt_output = np.array([float(val) for val in row[1:]]).reshape((1, TOTAL_CATEGORIES))
            else:
                gt_output = np.zeros((1, TOTAL_CATEGORIES))
            outputs.append(gt_output)
    outputs = np.array(outputs).reshape((-1, TOTAL_CATEGORIES))
    return inputs, outputs


def calculate_accruracy(y_true, y_pred):
    y_true_id = np.argmax(y_true, axis=1)
    y_pred_id = np.argmax(y_pred, axis=1)
    confusion_matrix = metrics.confusion_matrix(y_true_id, y_pred_id)
    print(confusion_matrix)
    accuracy = metrics.accuracy_score(y_true_id, y_pred_id)
    print(accuracy)


def output_results(inputs, outputs, output_csv_file_name, header=["image_id", "healthy", "multiple_diseases", "rust", "scab"]):
    with open(output_csv_file_name, "w", encoding='utf8') as fid:
        fid.write(",".join(header))
        for i, image_id in enumerate(inputs):
            fid.write("\n" + image_id + ',')
            fid.write(",".join(["{0:.4f}".format(val) for val in outputs[i, :]]))


def main():
    """Main function
    """
    # Load the model
    model = tensorflow.keras.models.load_model('keras_model.h5')
    input_csv_file_name = "test.csv"

    # load metadata
    with open('metadata.json') as fid:
        metadata = json.load(fid)

    # load image ids
    inputs, outputs = load_input(input_csv_file_name)

    # predict
    predictions = predict_all(model, inputs)

    # save output
    output_results(inputs, predictions, "out_" + input_csv_file_name)

    # Metrix
    calculate_accruracy(outputs, predictions)


if __name__ == "__main__":
    # download_dataset()
    main()
