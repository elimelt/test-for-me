import os
import logging
import re
import glob
from unittest.mock import patch, mock_open, MagicMock
import pytest
from src.main import TestGenerator, clean_test_content, parse_ignore_patterns, prompt_gemini

# Mock Gemini API Key for testing
MOCK_GEMINI_API_KEY = "test_api_key"

# Test data
TEST_FILE_CONTENT = "def add(a, b):\n    return a + b"
TEST_FILE_PATH = "test_file.py"
TEST_OUTPUT_DIR = "test_output"
TEST_FILE_NAME = "test_test_file.py"

@pytest.fixture
def test_generator():
    """Fixture for creating a TestGenerator instance with mock API key."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": MOCK_GEMINI_API_KEY}):
        generator = TestGenerator(api_key=MOCK_GEMINI_API_KEY)
        return generator

def test_prompt_gemini_success():
    """Test successful API call to Gemini."""
    mock_response = MagicMock()
    mock_response.text = "Test response"

    with patch("src.main.client.models.generate_content", return_value=mock_response) as mock_generate_content:
        response = prompt_gemini("Test prompt")
        assert response == "Test response"
        mock_generate_content.assert_called_once_with(
            model="gemini-2.0-flash",
            contents="Test prompt"
        )

def test_prompt_gemini_failure():
    """Test API call failure handling."""
    with patch("src.main.client.models.generate_content", side_effect=Exception("API Error")):
        with patch("builtins.print") as mock_print:
            response = prompt_gemini("Test prompt")
            assert response == ""
            mock_print.assert_called_with("Error calling Gemini API: API Error")

def test_testgenerator_initialization():
    """Test the initialization of the TestGenerator class."""
    with patch("logging.basicConfig") as mock_logging_config:
        test_gen = TestGenerator(api_key=MOCK_GEMINI_API_KEY, verbose=True, dry_run=True)
        assert test_gen.api_key == MOCK_GEMINI_API_KEY
        assert test_gen.dry_run == True
        assert test_gen.logger.name == "gemini_test_gen"

        # Verify logging was configured with the correct level
        mock_logging_config.assert_called_once()
        args, kwargs = mock_logging_config.call_args
        assert kwargs["level"] == logging.DEBUG

def test_validate_python_file_valid(test_generator):
    """Test validating a valid Python file."""
    with patch("os.path.isfile", return_value=True), \
         patch("os.path.basename", return_value="myfile.py"):
        assert test_generator.validate_python_file("myfile.py") == True

def test_validate_python_file_not_a_file(test_generator):
    """Test validating when the path is not a file."""
    with patch("os.path.isfile", return_value=False), \
         patch.object(test_generator.logger, "warning") as mock_warning:
        assert test_generator.validate_python_file("myfile.py") == False
        mock_warning.assert_called_once_with("Not a file: myfile.py")

def test_validate_python_file_not_python(test_generator):
    """Test validating when the file is not a Python file."""
    with patch("os.path.isfile", return_value=True), \
         patch("os.path.basename", return_value="myfile.txt"), \
         patch.object(test_generator.logger, "warning") as mock_warning:
        assert test_generator.validate_python_file("myfile.txt") == False
        mock_warning.assert_called_once_with("Not a Python file: myfile.txt")

def test_validate_python_file_is_test_file(test_generator):
    """Test validating when the file is already a test file."""
    with patch("os.path.isfile", return_value=True), \
         patch("os.path.basename", return_value="test_myfile.py"), \
         patch.object(test_generator.logger, "info") as mock_info:
        assert test_generator.validate_python_file("test_myfile.py") == False
        mock_info.assert_called_once_with("Skipping test file: test_myfile.py")

def test_find_python_files_directory_recursive(test_generator):
    """Test finding Python files in a directory recursively."""
    mock_files = ["file1.py", "file2.py", "test_file.py"]

    with patch("os.path.isdir", return_value=True), \
         patch("glob.glob", return_value=mock_files), \
         patch("os.path.basename", side_effect=lambda x: x), \
         patch.object(test_generator.logger, "debug") as mock_debug:
        files = test_generator.find_python_files("mydir", recursive=True)
        assert files == ["file1.py", "file2.py"]
        # Verify the test file was properly logged as skipped
        mock_debug.assert_called_with("Skipping test file: test_file.py")

def test_find_python_files_directory_non_recursive(test_generator):
    """Test finding Python files in a directory non-recursively."""
    mock_files = ["file1.py", "file2.py", "test_file.py"]

    with patch("os.path.isdir", return_value=True), \
         patch("glob.glob", return_value=mock_files), \
         patch("os.path.basename", side_effect=lambda x: x):
        files = test_generator.find_python_files("mydir", recursive=False)
        assert files == ["file1.py", "file2.py"]
        # Verify we're using the correct glob pattern without recursion
        glob.glob.assert_called_once_with(os.path.join("mydir", "*.py"), recursive=False)

def test_find_python_files_single_file_valid(test_generator):
    """Test finding a single valid Python file."""
    with patch("os.path.isdir", return_value=False), \
         patch.object(test_generator, "validate_python_file", return_value=True):
        files = test_generator.find_python_files("myfile.py")
        assert files == ["myfile.py"]
        # Verify validate_python_file was called with the correct path
        test_generator.validate_python_file.assert_called_once_with("myfile.py")

def test_find_python_files_single_file_invalid(test_generator):
    """Test finding a single invalid Python file."""
    with patch("os.path.isdir", return_value=False), \
         patch.object(test_generator, "validate_python_file", return_value=False):
        files = test_generator.find_python_files("myfile.py")
        assert files == []

def test_find_python_files_with_ignore_patterns(test_generator):
    """Test finding Python files with ignore patterns."""
    mock_files = ["file1.py", "file2.py", "excluded.py", "test_file.py"]
    ignore_patterns = [re.compile(re.escape("excluded.py"))]

    with patch("os.path.isdir", return_value=True), \
         patch("glob.glob", return_value=mock_files), \
         patch("os.path.basename", side_effect=lambda x: x), \
         patch.object(test_generator.logger, "debug") as mock_debug:
        files = test_generator.find_python_files("mydir", recursive=True, ignore_patterns=ignore_patterns)
        assert set(files) == set(["file1.py", "file2.py"])
        # Verify both skipped test file and ignored pattern were logged
        mock_debug.assert_any_call("Skipping test file: test_file.py")
        mock_debug.assert_any_call("Ignoring file (matched pattern): excluded.py")

def test_read_python_file_success(test_generator):
    """Test reading a Python file successfully."""
    mock_file_content = "def my_function():\n    pass"

    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        content = test_generator.read_python_file("myfile.py")
        assert content == mock_file_content

def test_read_python_file_failure(test_generator):
    """Test handling failure when reading a Python file."""
    with patch("builtins.open", side_effect=FileNotFoundError("File not found")), \
         patch.object(test_generator.logger, "error") as mock_error:
        content = test_generator.read_python_file("myfile.py")
        assert content == ""
        # Verify error was logged with the exception message
        mock_error.assert_called_once()
        assert "File not found" in str(mock_error.call_args)

def test_generate_test_prompt(test_generator):
    """Test generating a prompt for the Gemini API."""
    prompt = test_generator.generate_test_prompt(TEST_FILE_CONTENT, TEST_FILE_PATH)
    # Check all required components of the prompt
    assert TEST_FILE_PATH in prompt
    assert TEST_FILE_CONTENT in prompt
    assert "You are a Python test engineer" in prompt
    assert "pytest framework" in prompt
    assert "Use mocks and patches" in prompt
    assert "Cover edge cases" in prompt
    assert "Return only the test code" in prompt

def test_get_output_file_path_mirror(test_generator):
    """Test getting the output file path with mirroring."""
    with patch("os.path.basename", return_value="myfile.py"):
        output_path = test_generator.get_output_file_path("myfile.py", TEST_OUTPUT_DIR, mirror=True)
        assert output_path == os.path.join(TEST_OUTPUT_DIR, "test_myfile.py")

def test_get_output_file_path_no_mirror(test_generator):
    """Test getting the output file path without mirroring."""
    with patch("os.path.relpath", return_value="my/module/myfile.py"), \
         patch("os.path.dirname", return_value="my/module"), \
         patch("os.makedirs", return_value=None) as mock_makedirs, \
         patch("os.path.basename", return_value="myfile.py"):
        output_path = test_generator.get_output_file_path("myfile.py", TEST_OUTPUT_DIR, mirror=False)
        expected_path = os.path.join(TEST_OUTPUT_DIR, "my/module", "test_myfile.py")
        assert output_path == expected_path
        # Verify directories were created with exist_ok=True
        mock_makedirs.assert_called_once_with(os.path.join(TEST_OUTPUT_DIR, "my/module"), exist_ok=True)

def test_write_test_file_dry_run(test_generator):
    """Test writing the test file in dry run mode."""
    test_generator.dry_run = True

    with patch("os.path.exists", return_value=False), \
         patch("os.makedirs", return_value=None), \
         patch("builtins.open") as mock_file, \
         patch.object(test_generator.logger, "info") as mock_info:
        result = test_generator.write_test_file("output.py", "test content")
        assert result == True
        mock_file.assert_not_called()
        # Verify dry run message was logged
        mock_info.assert_called_once_with("Dry run mode - would write to: output.py")

def test_write_test_file_success(test_generator):
    """Test writing the test file successfully."""
    with patch("os.path.exists", return_value=False), \
         patch("os.makedirs", return_value=None), \
         patch("builtins.open", mock_open()) as mock_file, \
         patch.object(test_generator.logger, "info") as mock_info:
        result = test_generator.write_test_file("output.py", "test content")
        assert result == True
        mock_file.return_value.write.assert_called_once_with("test content")
        # Verify success message was logged
        mock_info.assert_called_once_with("Test file written to: output.py")

def test_write_test_file_already_exists_no_force(test_generator):
    """Test handling when the test file already exists and force is not enabled."""
    with patch("os.path.exists", return_value=True), \
         patch.object(test_generator.logger, "warning") as mock_warning:
        result = test_generator.write_test_file("output.py", "test content")
        assert result == False
        # Verify warning was logged
        mock_warning.assert_called_once_with("File already exists (use --force to overwrite): output.py")

def test_write_test_file_already_exists_with_force(test_generator):
    """Test handling when the test file already exists and force is enabled."""
    with patch("os.path.exists", return_value=True), \
         patch("os.makedirs", return_value=None), \
         patch("builtins.open", mock_open()) as mock_file:
        result = test_generator.write_test_file("output.py", "test content", force=True)
        assert result == True
        mock_file.return_value.write.assert_called_once_with("test content")

def test_write_test_file_error(test_generator):
    """Test handling an error when writing the test file."""
    error_message = "Write error"

    with patch("os.path.exists", return_value=False), \
         patch("os.makedirs", return_value=None), \
         patch("builtins.open", side_effect=IOError(error_message)), \
         patch.object(test_generator.logger, "error") as mock_error:
        result = test_generator.write_test_file("output.py", "test content")
        assert result == False
        # Verify error was logged with the exception message
        mock_error.assert_called_once()
        assert error_message in str(mock_error.call_args)

def test_generate_tests_for_file_success(test_generator):
    """Test generating tests for a file successfully."""
    with patch.object(test_generator, "read_python_file", return_value=TEST_FILE_CONTENT), \
         patch.object(test_generator, "generate_test_prompt", return_value="Test prompt"), \
         patch("src.main.prompt_gemini", return_value="Test content"), \
         patch.object(test_generator, "get_output_file_path", return_value="output.py"), \
         patch.object(test_generator, "write_test_file", return_value=True), \
         patch("src.main.clean_test_content", return_value="Cleaned test content"):
        result = test_generator.generate_tests_for_file(TEST_FILE_PATH, TEST_OUTPUT_DIR)
        assert result == True
        # Verify all methods were called with correct parameters
        test_generator.read_python_file.assert_called_once_with(TEST_FILE_PATH)
        test_generator.generate_test_prompt.assert_called_once_with(TEST_FILE_CONTENT, TEST_FILE_PATH)
        # Check that get_output_file_path was called with the correct positional arguments
        # and verify the call without using keyword arguments which can cause test failures
        args, kwargs = test_generator.get_output_file_path.call_args
        assert args[0] == TEST_FILE_PATH
        assert args[1] == TEST_OUTPUT_DIR
        assert kwargs.get('mirror', args[2] if len(args) > 2 else None) is False

def test_generate_tests_for_file_read_failure(test_generator):
    """Test handling when reading the Python file fails."""
    with patch.object(test_generator, "read_python_file", return_value=""), \
         patch.object(test_generator.logger, "info") as mock_info:
        result = test_generator.generate_tests_for_file(TEST_FILE_PATH, TEST_OUTPUT_DIR)
        assert result == False
        mock_info.assert_called_once_with(f"Generating tests for: {TEST_FILE_PATH}")

def test_generate_tests_for_file_prompt_gemini_failure(test_generator):
    """Test handling when the Gemini API call fails."""
    with patch.object(test_generator, "read_python_file", return_value=TEST_FILE_CONTENT), \
         patch.object(test_generator, "generate_test_prompt", return_value="Test prompt"), \
         patch("src.main.prompt_gemini", return_value=""), \
         patch.object(test_generator.logger, "error") as mock_error:
        result = test_generator.generate_tests_for_file(TEST_FILE_PATH, TEST_OUTPUT_DIR)
        assert result == False
        mock_error.assert_called_once_with(f"Failed to generate tests for: {TEST_FILE_PATH}")

def test_clean_test_content_with_code_blocks():
    """Test cleaning the test content by removing surrounding code blocks."""
    test_content = "```python\nThis is test code.\n```\nSome extra text."
    cleaned_content = clean_test_content(test_content)
    assert cleaned_content == "This is test code."

def test_clean_test_content_with_multiple_code_blocks():
    """Test cleaning content with multiple code blocks."""
    test_content = "```python\nFirst block\n```\nText in between\n```python\nSecond block\n```"
    cleaned_content = clean_test_content(test_content)
    assert cleaned_content == "First block\nSecond block"

def test_clean_test_content_no_code_blocks():
    """Test cleaning content without code blocks."""
    test_content = "This is test code.\nSome extra text."
    cleaned_content = clean_test_content(test_content)
    assert cleaned_content == ""

def test_clean_test_content_incomplete_code_blocks():
    """Test cleaning content with incomplete code blocks."""
    test_content = "```\nThis is test code."
    cleaned_content = clean_test_content(test_content)
    assert cleaned_content == ""

def test_parse_ignore_patterns_none():
    """Test parsing ignore patterns with None value."""
    patterns = parse_ignore_patterns(None)
    assert patterns == []

def test_parse_ignore_patterns_empty_string():
    """Test parsing ignore patterns with an empty string."""
    patterns = parse_ignore_patterns("")
    assert patterns == []

def test_parse_ignore_patterns_single_pattern():
    """Test parsing a single ignore pattern."""
    patterns = parse_ignore_patterns("*.txt")
    assert len(patterns) == 1
    assert isinstance(patterns[0], re.Pattern)
    # Test the pattern matches correctly
    assert patterns[0].match("file.txt") is not None
    assert patterns[0].match("file.py") is None

def test_parse_ignore_patterns_multiple_patterns():
    """Test parsing multiple ignore patterns."""
    patterns = parse_ignore_patterns("*.txt,*.log")
    assert len(patterns) == 2
    # Test both patterns match correctly
    assert patterns[0].match("file.txt") is not None
    assert patterns[1].match("file.log") is not None
    assert patterns[0].match("file.py") is None
    assert patterns[1].match("file.py") is None

def test_parse_ignore_patterns_with_whitespace():
    """Test parsing ignore patterns with whitespace."""
    patterns = parse_ignore_patterns(" *.txt ,  *.log ")
    assert len(patterns) == 2
    # Test both patterns match correctly with whitespace handled
    assert patterns[0].match("file.txt") is not None
    assert patterns[1].match("file.log") is not None
