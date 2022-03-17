

class Datapoint:
    def __init__(self):
        pass

    def __str__(self):
        var_dict = vars(self)
        dp_str = ""
        for key, value in var_dict.items():
            dp_str += "%s: %s\n" % (key, value)
        return dp_str
