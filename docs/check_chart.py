from imageai.Prediction.Custom import CustomImagePrediction

def create_prediction(json_path='../models/model_class.json', model_path='../models/model_ex-017_acc-0.937500.h5'):
    prediction = CustomImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(model_path)
    prediction.setJsonPath(json_path)
    prediction.loadModel(num_objects=10)

    return prediction


def check_image(prediction, image_name='selfie.png', path='../images/'):
    img_path = path + image_name
    predictions, probabilities = prediction.predictImage(img_path, result_count=3)

    print('Image ', image_name, ":")
    for prediction, probability in zip(predictions, probabilities):
        if prediction == 'not a chart':
            prediction = 'Таджик'
        print(prediction, " : ", probability, '%')
        if probability > 50:
            return str(prediction), str(probability)

def onCameraClick(self, *args):
        global predict_model
        self.cameraObject.export_to_png('../images/selfie.png')
        crop_image()
        result = check_image(prediction=predict_model)
        self.lbl_text = result[0] + ': ' + result[1] + '%'

if __name__ == '__main__':
    prediction = create_prediction()
    check_image(prediction)
