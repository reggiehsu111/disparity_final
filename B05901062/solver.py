from model import Model

class Solver():
    def __init__(self, img_left, img_right, disp_max):
        self.h, self.w, self.ch = img_left.shape
        self.img_left = img_left
        self.img_right = img_right
        self.disp_max = disp_max
        self.model = Model()
        self.model.load_model('model/cnn_e:13.pkl', 'model/fcnet_e:13.pkl')

    def compute_similarity(self):
        return self.model.calc_l_similarity(self.img_left, self.img_right, self.disp_max)
