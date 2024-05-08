import os

def move_files_to_parent(folder_path):
  """
  Moves all files from subfolders within a directory to the parent directory.

  Args:
      folder_path: The path to the directory containing subfolders.
  """
  for entry in os.scandir(folder_path):
    if entry.is_dir():  # Check if it's a directory
      subfolder_path = entry.path
      for subfile in os.listdir(subfolder_path):
        source = os.path.join(subfolder_path, subfile)
        destination = os.path.join(folder_path, subfile)
        os.rename(source, destination)

def are_all_folders_empty(folder_path):
  """
  Checks if all subfolders within a directory are empty.

  Args:
      folder_path: The path to the directory containing subfolders.

  Returns:
      True if all subfolders are empty, False otherwise.
  """
  for entry in os.scandir(folder_path):
    if entry.is_dir():  # Check if it's a directory
      subfolder_path = entry.path
      if any(os.scandir(subfolder_path)):  # Check if subfolder has any entries (files or folders)
        return False
  return True


def sort_by_place(data_path, sort_by):
  """
  Sorts data files in a directory by ID and puts them into folders named after the ID.

  Args:
      data_path: The path to the directory containing the data files.
  """
  for entry in os.scandir(data_path):
    if entry.is_file():  # Check if it's a file (not a folder)
      filename = entry.name
      # Extract ID assuming consistent format with leading digits
      id_ = filename.split("_")[sort_by]
      folder_path = os.path.join(data_path, id_)
      
      # Create folder if it doesn't exist
      if not os.path.exists(folder_path):
        os.makedirs(folder_path)
      
      # Move file to the corresponding folder
      source = os.path.join(data_path, filename)
      destination = os.path.join(folder_path, filename)
      os.rename(source, destination)

