from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from check_chart import check_image, create_prediction
from crop_black_background import crop_image
from graph_adt import GraphADT
from main import quantize, read_image
from kivy.uix.popup import Popup
import os

class StartScreen(Screen):
    def switchToNextScreen(self):
        global predict_model
        predict_model = create_prediction()
        self.parent.current = 'main_screen'


class PredictScreen(Screen):
    def save_to_gallery(self):
        print('os will copy to gallery dir')

    def update_prediction_image(self):
        self.ids.res_img.reload()

class PopupScreen(Screen):
    global graph

    def process_linear(self):
        graph.simple_reveal(save=True)

    def process_polynomial(self):
        graph.reveal(save=True, axes=False)

    def process_ai(self):
        print('Not implemented yet')


class ResultScreen(Screen):
    img_source = '../images/selfie.png'
    global graph

    def save_data(self):
        graph.to_csv(f_name='result.csv')
        print('Saved')

    def save_plot(self):
        graph.save_plot()

    def update_image(self):
        self.ids.selfie_img.reload()
        print('image  updating')

    def show_popup(self):
        show = PopupScreen()
        self.popupWindow = Popup(auto_dismiss=True, title='Choose prediction type', content=show, size_hint=(None, None), size=(400, 400))
        self.popupWindow.open()


class MainScreen(Screen):
    def onCameraClick(self, *args):
        global predict_model
        self.ids.camera.export_to_png('../images/selfie.png')
        # process-only function
        crop_image()

        result = check_image(prediction=predict_model)
        self.ids.imglbl.text = result[0] + ': ' + result[1] + '%'
        print(result[0])
        if result[0].lower() == 'chart' or True:
            self.parent.current = 'result_screen'
            img = read_image('../images/selfie.png')
            points = quantize(img)
            global graph
            graph = GraphADT(points)
            return True
        return False

    def load_file(self):
        print('loading file')


class ScreenMan(ScreenManager):
    pass


class TestApp(App):

    def build(self):
        self.sm = ScreenMan()
        self.ss = StartScreen(name='start_screen')
        self.rs = ResultScreen(name='result_screen')
        self.ps = PredictScreen(name='predict_screen')
        self.popup = PopupScreen(name='popup')

        self.title = 'PhotoGraph!'
        # start screen only called while loadin program
        self.sm.add_widget(self.ss)
        self.sm.add_widget(MainScreen(name='main_screen'))
        self.sm.add_widget(self.rs)
        self.sm.add_widget(self.ps)
        self.sm.add_widget(self.popup)
        print(self.popup)
        return self.sm


if __name__ == '__main__':
    global predict_model
    global graph
    TestApp().run()
