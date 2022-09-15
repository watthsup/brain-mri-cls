import configparser

class GetDict:  
    def __init__(self, config):
        self.config = config
    
    def get_dict(self):
        config = configparser.SafeConfigParser()
        config.read(self.config)
        config_list =[]
        for section in config.sections():
            for item in config.items(section):
                config_list.append(item)
        return dict(config_list)

getdict = GetDict('config.ini')
config = getdict.get_dict()
