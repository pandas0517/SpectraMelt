class NYFR:
    def __init__(self, system_config):
        self.system_name = system_config['system_name']
        self.filter = system_config['filter']
        self.adc_clock_freq = system_config['adc_clock_freq']
        self.start = system_config['start']
        self.stop = system_config['stop']
        self.spacing = system_config['spacing']
        self.noise = system_config['noise']
        self.processing_systems = system_config['processing_systems']
        self.LO_params = system_config['LO_params']