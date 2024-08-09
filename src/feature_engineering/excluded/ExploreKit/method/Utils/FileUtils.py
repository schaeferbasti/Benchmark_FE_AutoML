from os import listdir, remove, rename
from os.path import isdir, isfile, join, abspath, split

from src.feature_engineering.excluded.ExploreKit.method.Utils.Logger import Logger
'''
return list of the files in the directory
if there's no files, return empty list
if the dir doesn't exit, return None
'''
def listFilesInDir(directory: str) -> list:
    if not isdir(directory):
        return None
    files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
    return files

def deleteFile(filePath: str):
    try:
        remove(filePath)
    except:
        Logger.Error('Failed delete file: ' + filePath)

def getAbsPath(filePath: str):
    try:
        return abspath(filePath)
    except:
        Logger.Error('Failed get file abs path for ' + filePath)

def getFilenameFromPath(filePath: str) -> str:
    try:
        folderPath, fileName = split(filePath)
        return fileName
    except:
        Logger.Error('Failed get filename out of file path: ' + filePath)

def renameFile(oldFilePath: str, newFilePath: str):
    rename(r'{}'.format(oldFilePath), r'{}'.format(newFilePath))

def isFileExist(filePath: str) -> bool:
    return isfile(filePath)