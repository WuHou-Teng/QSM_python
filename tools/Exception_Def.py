class ParamException(Exception):
    def __init__(self, area, situation):
        super(ParamException, self).__init__()
        self.area = area
        self.situation = situation

    def get_self_name(self):
        return str(type(self)).split('.')[1][:-3]

    def info(self):
        return self.area + ": " + self.get_self_name() + ": " + self.situation
