# Model's arguments

class ModelArguments:
    def __init__(self, 
                 path_to_data: str, 
                 path_to_annotation: str, 
                 signals_names: list[str], 
                 use_spectrum: bool = False
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
    def number_of_channels(self) -> int:
        """
        Get the total number of channels (including time).

        Returns:
        - int: Total number of channels.
        """
        return len(self.signals_names) * 2 + 1 \
                if self.use_spectrum else len(self.signals_names) + 1

