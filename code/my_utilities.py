import os
from datetime import datetime
import errno

def filecreation(filepath, filename='gru'):
  mydir = os.path.join(
    os.getcwd(),
    filepath,
    datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '@' + filename
  )
  try:
    os.makedirs(mydir)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise  # This was not a "directory exist" error..

  return mydir
