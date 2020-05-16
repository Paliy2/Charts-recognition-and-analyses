from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from crop_black_background import crop_image
from graph_adt import GraphADT
from process import quantize, read_image
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from adt import total_processing
from image_adt import Image_ADT

class StartScreen(Screen):
    def switchToNextScreen(self):
        # global predict_model
        # predict_model = create_prediction()
        self.parent.current = 'main_screen'


class PredictScreen(Screen):
    def save_to_gallery(self):
        print('os will copy to gallery dir')

    def update_prediction_image(self):
        self.ids.res_img.reload()


class DirScreen(Screen):
    pass


class PopupScreen(Screen):
    global graph

    def process_linear(self):
        graph.simple_reveal()

    def process_polynomial(self):
        graph.reveal(axes=False)

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
        print('image updating')

    def show_popup(self):
        show = PopupScreen()
        self.popupWindow = Popup(auto_dismiss=True, title='Choose prediction type', content=show, size_hint=(.8, .7))
        self.popupWindow.open()


class MainScreen(Screen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_graphs = 1
        self.dir = ''
    def onCameraClick(self, *args):
        try:
            img = read_image(self.dir)
            self.parent.rs.update_image()
            path = self.dir
            text = 'File has been read'
        except:
            self.ids.camera.export_to_png('../images/selfie.png')
            # process-only function
            crop_image()
            img = read_image('../images/selfie.png')
            path = '../images/selfie.png'
            text = "Great! Photo will Graph soon"
        self.ids.imglbl.text = text
        self.parent.current = 'result_screen'
        # points = quantize(img)
        # to work properly add 1 for background
        img_class = Image_ADT(img)
        points = img_class.analyze(img)
        # points = total_processing(path, self.n_graphs + 1)
        global graph
        graph = GraphADT(points)
        graph.show()
        self.dir = ''
        return True

    def load_file(self):
        show = DirScreen()
        self.popupWindow = Popup(auto_dismiss=True, title='Enter full image directory', content=show,
                                 size_hint=(.8, .6))
        self.popupWindow.open()
        self.ok = True

    def get_graph_count(self, number):
        self.n_graphs = number

    def leave_input(self, name):
        print('NAME:', name)
        # self.popupWindow.dismiss()
        self.dir = name
        try:
            img = read_image(self.dir)
            self.onCameraClick()
        except:
            self.ids.imglbl.text = "Can't open file: wrong directory"


class ScreenMan(ScreenManager):
    pass


class TestApp(App):

    def build(self):
        self.sm = ScreenMan()
        self.ss = StartScreen(name='start_screen')
        self.rs = ResultScreen(name='result_screen')
        self.ps = PredictScreen(name='predict_screen')
        self.popup = PopupScreen(name='popup')
        self.ds = DirScreen(name='dir_screen')
        self.ms = MainScreen(name='main_screen')
        self.title = 'PhotoGraph!'
        # start screen only called while loadin program
        self.sm.add_widget(self.ss)
        self.sm.add_widget(self.ds)
        self.sm.add_widget(self.ms)
        self.sm.add_widget(self.rs)
        self.sm.add_widget(self.ps)
        self.sm.add_widget(self.popup)
        return self.sm


if __name__ == '__main__':
    global graph
    TestApp().run()
