def load_data(file, config):
    """
    Load song data from a file.

    Parameters:
    - file (str): The path to the data file
    - config (dict): A dictionary containing configuration parameters:
        - "sequence_size" (int): The size of the sequences to extract from the data and train our model. 
        (We want to ensure that the song we use is at least as big as the sequence size)

    Returns:
    - data (list): A list of sequences extracted from the data file
    """

    # Extract configuration parameters
    SEQ_SIZE = config["sequence_size"]

    # Initialize an empty list to store the data
    data = []

    # Read data from the file
    with open(file, "r") as f:
        string_of_all_texts = f.readlines()[0]
        return string_of_all_texts.split()
    
    
    
    
    
#     with open(file, "r") as f:
#         buffer = ''
#         for line in f:
#             if line == '<start>\n':
#                 buffer = line
#             elif line == '<end>\n':
#                 buffer += line
#                 # Check if the buffer size meets the sequence size threshold
#                 if len(buffer) > SEQ_SIZE + 10:
#                     data.append(buffer)
#                 buffer = ''
#             else:
#                 buffer += line

#     return data