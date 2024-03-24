# Model's arguments

class ModelArguments:
    def __init__(self, 
                 path_to_data: str, 
                 path_to_annotation: str, 
                 signals_names: list[str], 
                 use_spectrum: bool=False,
                 use_class_balance: bool=True,
                 window_size: int=4,
                 sample_rate: int=128,
                 use_raw_data: bool=False,
                 number_of_extracting_features: int=3,
                ):
        """
        Initializes the ModelsArguments with the given parameters.

        Parameters:
        - path_to_data (str): Path to the data directory.
        - path_to_annotation (str): Path to the annotation file.
        - signals_names (list[str]): List of signal names.
        - use_spectrum (bool): Flag indicating whether to use spectrum (default: False).
        """
        self._path_to_data = path_to_data
        self._path_to_annotation = path_to_annotation
        self._signals_names = signals_names
        self._use_spectrum = use_spectrum
        self._use_class_balance = use_class_balance
        self._window_size = window_size
        self._sample_rate = sample_rate
        self._use_raw_data = use_raw_data
        self._number_of_extracting_features = number_of_extracting_features
        
    @property    
    def path_to_data(self) -> str:
        """
        Get the path to the data.

        Returns:
        - str: Path to the data.
        """
        return self._path_to_data
    
    @path_to_data.setter
    def path_to_data(self, path: str) -> None:
        """
        Set the path to the data.

        Parameters:
        - path (str): Path to the data.
        """
        if path is None:
            raise ValueError('Specify a path')
        self._path_to_data = path
        
    @property    
    def path_to_annotation(self) -> str:
        """
        Get the path to the annotation file.

        Returns:
        - str: Path to the annotation file.
        """
        return self._path_to_annotation
    
    @path_to_annotation.setter
    def path_to_annotation(self, path: str) -> None:
        """
        Set the path to the annotation file.

        Parameters:
        - path (str): Path to the annotation file.
        """
        if path is None:
            raise ValueError('Specify a path')
        self._path_to_annotation = path
        
    @property    
    def use_spectrum(self) -> bool:
        """
        Get the flag indicating whether to use spectrum.

        Returns:
        - bool: Flag indicating whether to use spectrum.
        """
        return self._use_spectrum
    
    @use_spectrum.setter
    def use_spectrum(self, use_spectrum: bool) -> None:
        """
        Set the flag indicating whether to use spectrum.

        Parameters:
        - use_spectrum (bool): Flag indicating whether to use spectrum.
        """
        self._use_spectrum = use_spectrum
        
    @property    
    def signals_names(self) -> list[str]:
        """
        Get the list of signal names.

        Returns:
        - list[str]: List of signal names.
        """
        return self._signals_names
    
    @signals_names.setter
    def signals_names(self, signals_names: list[str]) -> None:
        """
        Set the list of signal names.

        Parameters:
        - signals_names (list[str]): List of signal names.
        """
        if len(signals_names) == 0:
            raise ValueError('List of signals shouldn\'t be empty')
        self._signals_names = signals_names
        
    @property    
    def number_of_extracting_features(self) -> int:
        """
        Get the number of extracting features.

        Returns:
        - int: Number of extracting features.
        """
        return self._number_of_extracting_features
    
    @number_of_extracting_features.setter
    def number_of_extracting_features(self, number: int) -> None:
        """
        Set the  number of extracting features.

        Parameters:
        - number (int): Flag indicating whether to use raw data.
        """
        if number <= 0:
            raise Exception('Number of featured should be greater than 0')
        self._number_of_extracting_features = number
        
    @property    
    def number_of_channels(self) -> int:
        """
        Get the total number of channels (including time).

        Returns:
        - int: Total number of channels.
        """
        number_of_channels = len(self.signals_names) * 2 \
                if self.use_spectrum else len(self.signals_names)
        
        if self.use_raw_data is False:
            number_of_channels = number_of_channels * self.number_of_extracting_features
            
        # A channel with time    
        number_of_channels += 1
 
        return number_of_channels
    
    @property    
    def use_class_balance(self) -> int:
        """
        Get the flag indicating whether to use class balance.

        Returns:
        - int: Flag indicating whether to use class balance.
        """
        return self._use_class_balance
    
    @use_class_balance.setter
    def use_class_balance(self, use_balancing: bool) -> None:
        """
        Set the flag indicating whether to use class balance.

        Parameters:
        - use_balancing (bool): Flag indicating whether to use class balance.
        """
        self._use_class_balance = use_balancing
        
    @property    
    def sample_rate(self) -> int:
        """
        Get the sample rate.

        Returns:
        - int: Sample rate.
        """
        return self._sample_rate
    
    @sample_rate.setter
    def sample_rate(self, sample_rate: bool) -> None:
        """
        Set the sample rate.

        Parameters:
        - sample_rate (bool): Sample rate.
        """
        if sample_rate <= 0:
            raise Exception('Sample rate should be a positive number')
        self._sample_rate = sample_rate
                             
    @property    
    def window_size(self) -> int:
        """
        Get the window size.

        Returns:
        - int: Window size.
        """
        return self._window_size
    
    @window_size.setter
    def window_size(self, window_size: int) -> None:
        """
        Set the window size.

        Parameters:
        - window_size (int): Window size.
        """
        if window_size <= 0:
            raise Exception('Window_size should be a positive number')
        self._window_size = window_size
                             
    @property    
    def window_size_in_elements(self) -> int:
        """
        Get the window size in elements.

        Returns:
        - int: Window size in elements.
        """
        return self._window_size * self._sample_rate
    
    @property    
    def use_raw_data(self) -> int:
        """
        Get the flag indicating whether to use raw data.

        Returns:
        - int: Flag indicating whether to use raw data.
        """
        return self._use_raw_data
    
    @use_raw_data.setter
    def use_raw_data(self, use_raw_data: bool) -> None:
        """
        Set the flag indicating whether to use raw data.

        Parameters:
        - use_raw_data (bool): Flag indicating whether to use raw data.
        """
        self._use_raw_data = use_raw_data
        


