from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from crop_black_background import crop_image
from graph_adt import GraphADT, Multigraph
from kivy.uix.popup import Popup
from adt_new import ImageADT
from process import read_image, save_image
import time


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_graphs = 1
        self.im_dirs = ['../images/selfie.png', 'color_blobs.png', 'edges.png']

    def save_data(self):
        graph.to_csv(f_name='result.csv')
        print('Saved')

    def next_img(self):
        id = self.im_dirs.index(self.ids.selfie_img.source)
        try:
            return self.im_dirs[id + 1]
        except:
            return self.im_dirs[0]

    def prev_img(self):
        id = self.im_dirs.index(self.ids.selfie_img.source)
        try:
            return self.im_dirs[id - 1]
        except:
            return self.im_dirs[0]

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
        if not self.ok:
            print('here ok ne ok')
            self.ids.camera.export_to_png('../images/selfie.png')
            # process-only function
            crop_image()
            print('cropped, analyzing')
        # time.sleep(.3)
        print('IMAGEADT:')
        img = ImageADT('../images/selfie.png')
        ResultScreen().update_image()
        print('img updated')
        self.parent.current = 'result_screen'

        # to work properly add 1 for background
        graphs = img.analyze(self.n_graphs + 1)
        # points = img.analyze(img)
        # points = total_processing(path, self.n_graphs + 1)
        global graph
        # graph = Multigraph(graphs)
        print('global')
        graph = GraphADT(graphs[-1])
        # graph.show()
        print('Graphs has benn initialized')
        self.dir = ''
        return True

    def load_file(self):
        show = DirScreen()
        self.popupWindow = Popup(auto_dismiss=True, title='Enter full or relative image directory', content=show,
                                 size_hint=(.8, .6))
        self.popupWindow.open()
        self.ok = True

    def get_graph_count(self, number):
        self.n_graphs = number

    def leave_input(self, name):
        print('NAME:', name)
        # self.popupWindow.dismiss()
        self.dir = name.strip().replace('\\', '/')
        ok = False
        try:
            img = read_image(self.dir)
            save_image(img)
            ok = True
        except:
            print('wrong dir')
            self.ids.imglbl.text = "Can't open file: wrong directory"
        if ok:
            self.onCameraClick()


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
