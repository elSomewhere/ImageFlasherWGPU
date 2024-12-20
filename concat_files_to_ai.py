import os
import sys
import argparse

def collect_file_contents(paths, extensions):
    collected_text = ""
    processed_files = set()  # To avoid processing the same file multiple times

    for path in paths:
        if os.path.isfile(path):
            if path.lower().endswith(extensions):
                absolute_path = os.path.abspath(path)
                if absolute_path not in processed_files:
                    file_content = read_file_content(absolute_path)
                    if file_content is not None:
                        separator = f"\n{'*' * 21}\n{os.path.basename(absolute_path)}\n{'*' * 21}\n\n"
                        collected_text += separator + file_content + "\n\n"
                        processed_files.add(absolute_path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(extensions):
                        file_path = os.path.join(root, file)
                        absolute_path = os.path.abspath(file_path)
                        if absolute_path not in processed_files:
                            file_content = read_file_content(absolute_path)
                            if file_content is not None:
                                # Get relative path from the folder argument for better readability
                                relative_path = os.path.relpath(file_path, start=path)
                                separator = f"\n{'*' * 21}\n{relative_path}\n{'*' * 21}\n\n"
                                collected_text += separator + file_content + "\n\n"
                                processed_files.add(absolute_path)
        else:
            print(f"Warning: The path '{path}' is neither a file nor a directory and will be skipped.")

    return collected_text

def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading '{file_path}': {e}")
            return None
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
        return None

def write_to_file(output_path, content):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Contents written to '{output_path}' successfully!")
    except Exception as e:
        print(f"Failed to write to '{output_path}': {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Concatenate contents of specified .h, .cpp, and .py files or all such files within specified folders and write to "for_ai.txt".'
    )
    parser.add_argument(
        'paths',
        type=str,
        nargs='+',
        help='One or more file and/or folder paths to process.'
    )

    args = parser.parse_args()
    input_paths = args.paths

    # Validate paths
    valid_paths = []
    for path in input_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            print(f"Warning: The path '{path}' does not exist and will be skipped.")

    if not valid_paths:
        print("Error: No valid files or directories provided.")
        sys.exit(1)

    extensions = ('.h', '.cpp', '.py')
    print(f"Processing the following paths:")
    for p in valid_paths:
        print(f" - {p}")
    print(f"Looking for files with extensions: {', '.join(extensions)}...\n")

    collected_text = collect_file_contents(valid_paths, extensions)

    if not collected_text.strip():
        print("No files found with the specified extensions.")
        sys.exit(0)

    # Determine the script's directory to place for_ai.txt there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, 'for_ai.txt')

    write_to_file(output_file, collected_text)

if __name__ == "__main__":
    main()
