# Nome: Filippo
# Cognome: Morello
# Matricola: SM3201475

# Eccezione base per tutti gli errori del motore di rendering
class renderingEngineException(Exception):
    pass

# Errore sollevato se il file della palette non è valido
class invalidPaletteException(renderingEngineException):
    pass

# Errore sollevato se il file binario delle texture/sprite non è corretto
class invalidVramException(renderingEngineException):
    pass

# Errore sollevato se il JSON della scena contiene valori errati o mancanti
class invalidSceneException(renderingEngineException):
    pass

# Errore sollevato per fallimenti generici durante il disegno o salvataggio
class renderingException(renderingEngineException):
    pass
