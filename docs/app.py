from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

from check_chart import check_image, create_prediction
from crop_black_background import crop_image


class MyTest(BoxLayout):
    def onCameraClick(self, *args):
        global predict_model
        self.ids.camera.export_to_png('../images/selfie.png')
        crop_image()
        result = check_image(prediction=predict_model)
        self.ids.imglbl.text = result[0] + ': ' + result[1] + '%'

    def load_file(self):
        print('loading file')


class TestApp(App):
    def build(self):
        return MyTest()


if __name__ == '__main__':
    global predict_model
    predict_model = create_prediction()
    TestApp().run()
