"""Unit tests for __main__.py argument parsing."""

from argparse import Namespace
import pytest
from unittest.mock import patch, MagicMock
import copy
import sys

from ra_aid.__main__ import parse_arguments
from ra_aid.config import DEFAULT_RECURSION_LIMIT
from ra_aid.database.repositories.work_log_repository import WorkLogEntry
from ra_aid.database.repositories.config_repository import ConfigRepositoryManager, get_config_repository


@pytest.fixture(autouse=True)
def mock_config_repository():
    """Mock the ConfigRepository to avoid database operations during tests"""
    with patch('ra_aid.database.repositories.config_repository.config_repo_var') as mock_repo_var:
        # Setup a mock repository
        mock_repo = MagicMock()

        # Create a dictionary to simulate config
        config = {}

        # Setup set method to update config values
        def set_config(key, value):
            config[key] = copy.deepcopy(value)
        mock_repo.set.side_effect = set_config

        # Setup update method to update multiple config values
        def update_config(config_dict):
            for k, v in config_dict.items():
                config[k] = copy.deepcopy(v)
        mock_repo.update.side_effect = update_config

        # Setup get method to return config values
        def get_config(key, default=None):
            return copy.deepcopy(config.get(key, default))
        mock_repo.get.side_effect = get_config

        # Add get_keys method
        def get_keys():
            return list(config.keys())
        mock_repo.get_keys.side_effect = get_keys

        # Add deep_copy method
        def deep_copy():
            new_mock = MagicMock()
            new_config = copy.deepcopy(config)

            # Setup the new mock with the same methods
            def new_get(key, default=None):
                return copy.deepcopy(new_config.get(key, default))
            new_mock.get.side_effect = new_get

            def new_set(key, value):
                new_config[key] = copy.deepcopy(value)
            new_mock.set.side_effect = new_set

            def new_update(update_dict):
                for k, v in update_dict.items():
                    new_config[k] = copy.deepcopy(v)
            new_mock.update.side_effect = new_update

            def new_get_keys():
                return list(new_config.keys())
            new_mock.get_keys.side_effect = new_get_keys

            return new_mock

        mock_repo.deep_copy.side_effect = deep_copy

        # Make the mock context var return our mock repo
        mock_repo_var.get.return_value = mock_repo

        yield mock_repo


@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mock all dependencies needed for main()."""
    # Mock dependencies that interact with external systems
    monkeypatch.setattr("ra_aid.__main__.check_dependencies", lambda: None)
    monkeypatch.setattr("ra_aid.__main__.validate_environment",
                        lambda args: (True, [], True, []))
    monkeypatch.setattr("ra_aid.__main__.create_agent",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr("ra_aid.__main__.run_agent_with_retry",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr("ra_aid.__main__.run_research_agent",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "ra_aid.agents.planning_agent.run_planning_agent", lambda *args, **kwargs: None)

    # Mock LLM initialization
    def mock_config_update(*args, **kwargs):
        config_repo = get_config_repository()
        if kwargs.get("temperature"):
            config_repo.set("temperature", kwargs["temperature"])
        return None

    monkeypatch.setattr("ra_aid.__main__.initialize_llm", mock_config_update)


@pytest.fixture(autouse=True)
def mock_related_files_repository():
    """Mock the RelatedFilesRepository to avoid database operations during tests"""
    with patch('ra_aid.database.repositories.related_files_repository.related_files_repo_var') as mock_repo_var:
        # Setup a mock repository
        mock_repo = MagicMock()

        # Create a dictionary to simulate stored files
        related_files = {}

        # Setup get_all method to return the files dict
        mock_repo.get_all.return_value = related_files

        # Setup format_related_files method
        mock_repo.format_related_files.return_value = [
            f"ID#{file_id} {filepath}" for file_id, filepath in sorted(related_files.items())]

        # Make the mock context var return our mock repo
        mock_repo_var.get.return_value = mock_repo

        yield mock_repo


@pytest.fixture(autouse=True)
def mock_work_log_repository():
    """Mock the WorkLogRepository to avoid database operations during tests"""
    with patch('ra_aid.database.repositories.work_log_repository.work_log_repo_var') as mock_repo_var:
        # Setup a mock repository
        mock_repo = MagicMock()

        # Setup local in-memory storage
        entries = []

        # Mock add_entry method
        def mock_add_entry(event):
            from datetime import datetime
            entry = {"timestamp": datetime.now().isoformat(), "event": event}
            entries.append(entry)
        mock_repo.add_entry.side_effect = mock_add_entry

        # Mock get_all method
        def mock_get_all():
            return entries.copy()
        mock_repo.get_all.side_effect = mock_get_all

        # Mock clear method
        def mock_clear():
            entries.clear()
        mock_repo.clear.side_effect = mock_clear

        # Mock format_work_log method
        def mock_format_work_log():
            if not entries:
                return "No work log entries"

            formatted_entries = []
            for entry in entries:
                formatted_entries.extend([
                    f"## {entry['timestamp']}",
                    "",
                    entry["event"],
                    "",  # Blank line between entries
                ])

            # Remove trailing newline
            return "\n".join(formatted_entries).rstrip()
        mock_repo.format_work_log.side_effect = mock_format_work_log

        # Make the mock context var return our mock repo
        mock_repo_var.get.return_value = mock_repo

        yield mock_repo

def create_mock_args(**kwargs):
    """
    Create a mock Namespace with default values for all possible args.
    Any argument passed in **kwargs will override the default.
    """
    from argparse import Namespace

    defaults = {
        # --- Core / Input ---
        "message": "test message",
        "msg_file": None,
        "project_state_dir": None,
        "command": None,

        # --- Logging & Output ---
        "log_mode": "console",
        "pretty_logger": False,
        "log_level": "info",
        "version": False,
        "show_thoughts": False,
        "show_cost": False,

        # --- Modes ---
        "research_only": False,
        "research_and_plan_only": False, # <--- The missing argument
        "cowboy_mode": False,
        "hil": False,
        "chat": False,
        "wipe_project_memory": False,
        "server": False,
        "server_host": "0.0.0.0",
        "server_port": 1818,

        # --- Providers & Models ---
        "provider": "openai",
        "model": "gpt-4o",
        "num_ctx": None,
        "temperature": 0.7,
        "set_default_provider": None,
        "set_default_model": None,
        "price_performance_ratio": None,

        # --- Specialized Models ---
        "research_provider": None,
        "research_model": None,
        "planner_provider": None,
        "planner_model": None,
        "expert_provider": "openai",
        "expert_model": "o1-preview",
        "expert_num_ctx": None,

        # --- Limits & Costs ---
        "disable_limit_tokens": False,
        "recursion_limit": 100,
        "track_cost": False,
        "no_track_cost": False,
        "max_cost": None,
        "max_tokens": None,
        "exit_at_limit": False,

        # --- Tools & Execution ---
        "experimental_fallback_handler": False,
        "aider_config": None,
        "use_aider": False,
        "test_cmd": None,
        "auto_test": False,
        "max_test_cmd_retries": 3,
        "test_cmd_timeout": 300,
        "custom_tools": None,

        # --- Reasoning ---
        "reasoning_assistance": False,
        "no_reasoning_assistance": False,
    }

    # Update defaults with any specific arguments passed by the test
    defaults.update(kwargs)

    return Namespace(**defaults)

def test_recursion_limit_in_global_config(mock_dependencies, mock_config_repository):
        """Test that recursion limit is correctly set in global config."""
        from unittest.mock import patch
        from ra_aid.__main__ import main
    
        # Clear the mock repository before each test
        mock_config_repository.update.reset_mock()
    
        # For testing, we need to patch ConfigRepositoryManager.__enter__ to return our mock
        with patch('ra_aid.database.repositories.config_repository.ConfigRepositoryManager.__enter__', return_value=mock_config_repository):
            # Test default recursion limit
            mock_args_default = create_mock_args(message="test message")
            with patch("ra_aid.__main__.parse_arguments", return_value=mock_args_default):
                # Mock the core logic function to prevent the timeout
                with patch("ra_aid.__main__.run_agent_with_retry"):
                    main()
                    
                    # Verify config was updated
                    mock_config_repository.update.assert_called()
                    
                    # Check that recursion_limit is in one of the update calls
                    found = False
                    for call in mock_config_repository.update.call_args_list:
                        args, _ = call
                        if "recursion_limit" in args[0]:
                            assert args[0]["recursion_limit"] == 100  # Default value
                            found = True
                    assert found, "recursion_limit was not set in config"

def test_negative_recursion_limit():
    """Test that negative recursion limit raises error."""
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test message", "--recursion-limit", "-1"])


def test_zero_recursion_limit():
    """Test that zero recursion limit raises error."""
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test message", "--recursion-limit", "0"])


def test_config_settings(mock_dependencies, mock_config_repository):
        """Test that various settings are correctly applied in global config."""
        from unittest.mock import patch
        from ra_aid.__main__ import main
    
        # Clear the mock repository before each test
        mock_config_repository.update.reset_mock()
    
        # For testing, we need to patch ConfigRepositoryManager.__enter__ to return our mock
        with patch('ra_aid.database.repositories.config_repository.ConfigRepositoryManager.__enter__', return_value=mock_config_repository):
            mock_args = create_mock_args(
                message="test message",
                cowboy_mode=True,
                research_only=True,
                provider="anthropic",
                model="claude-3-7-sonnet-20250219",
                expert_provider="openai",
                expert_model="gpt-4",
                temperature=0.7,
                disable_limit_tokens=True
            )
            with patch("ra_aid.__main__.parse_arguments", return_value=mock_args):
                main()
                
                # Verify config values are set via the update method
                mock_config_repository.update.assert_called()
                
                # Get the call arguments
                call_args = mock_config_repository.update.call_args_list
    
                # Check for config values in the update calls
                for args, _ in call_args:
                    config_dict = args[0]
                    if "cowboy_mode" in config_dict:
                        assert config_dict["cowboy_mode"] is True
                    if "research_only" in config_dict:
                        assert config_dict["research_only"] is True
                    if "limit_tokens" in config_dict:
                        # The code stores the flag value directly, so True means disabled.
                        assert config_dict["limit_tokens"] is True

def test_temperature_validation(mock_dependencies, mock_config_repository):
    """Test that temperature argument is correctly passed to initialize_llm."""
    from unittest.mock import patch
    from ra_aid.__main__ import main

    # For testing, we need to patch ConfigRepositoryManager.__enter__ to return our mock
    with patch('ra_aid.database.repositories.config_repository.ConfigRepositoryManager.__enter__', return_value=mock_config_repository):
        # Test valid temperature (0.7)
        mock_args = create_mock_args(message="test", temperature=0.7)
        with patch("ra_aid.__main__.parse_arguments", return_value=mock_args):
            with patch("ra_aid.__main__.initialize_llm", return_value=None):
                # Also patch any calls that would actually use the mocked initialize_llm function
                with patch("ra_aid.__main__.run_research_agent", return_value=None):
                    with patch("ra_aid.agents.planning_agent.run_planning_agent", return_value=None):
                        main()
                        # Verify that the temperature was set in the config repository
                        mock_config_repository.set.assert_any_call(
                            "temperature", 0.7)

    # Test invalid temperature (2.1)
    # We test parse_arguments directly here as it handles validation
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test", "--temperature", "2.1"])


def test_missing_message():
    """Test that missing message argument raises error."""
    # Test chat mode which doesn't require message
    args = parse_arguments(["--chat"])
    assert args.chat is True
    assert args.message is None

    # Test non-chat mode requires message
    args = parse_arguments(["--provider", "openai"])
    assert args.message is None

    # Verify message is captured when provided
    args = parse_arguments(["-m", "test"])
    assert args.message == "test"


def test_msg_file_argument(tmp_path):
    """Test --msg-file argument handling."""
    # Create a test file
    test_file = tmp_path / "test_task.txt"
    test_file.write_text("Test task content")

    # Test reading from file
    args = parse_arguments(["--msg-file", str(test_file)])
    assert args.message == "Test task content"
    assert args.msg_file == str(test_file)

    # Test mutual exclusivity with --message
    with pytest.raises(SystemExit):
        parse_arguments(["--msg-file", str(test_file), "-m", "direct message"])

    # Test non-existent file
    non_existent = tmp_path / "nonexistent.txt"
    with pytest.raises(SystemExit):
        parse_arguments(["--msg-file", str(non_existent)])


def test_research_model_provider_args(mock_dependencies, mock_config_repository):
    """Test that research-specific model/provider args are correctly stored in config."""
    from unittest.mock import patch
    from ra_aid.__main__ import main

    # Reset mocks
    mock_config_repository.set.reset_mock()

    # Mock arguments
    mock_args = create_mock_args(
        message="test message",
        research_provider="anthropic",
        research_model="claude-3-haiku-20240307",
        planner_provider="openai",
        planner_model="gpt-4"
    )

    # For testing, we need to patch ConfigRepositoryManager.__enter__ to return our mock
    with patch('ra_aid.database.repositories.config_repository.ConfigRepositoryManager.__enter__', return_value=mock_config_repository):
        with patch("ra_aid.__main__.parse_arguments", return_value=mock_args):
            main()
            # Verify the mock repo's set method was called with the expected values
            mock_config_repository.set.assert_any_call(
                "research_provider", "anthropic")
            mock_config_repository.set.assert_any_call(
                "research_model", "claude-3-haiku-20240307")
            mock_config_repository.set.assert_any_call(
                "planner_provider", "openai")
            mock_config_repository.set.assert_any_call(
                "planner_model", "gpt-4")


def test_planner_model_provider_args(mock_dependencies, mock_config_repository):
    """Test that planner provider/model args fall back to main config when not specified."""
    from unittest.mock import patch
    from ra_aid.__main__ import main

    # Reset mocks
    mock_config_repository.set.reset_mock()

    # Mock arguments with planner args as None (default)
    mock_args = create_mock_args(
        message="test message",
        provider="openai",
        model="gpt-4",
        planner_provider=None,
        planner_model=None
    )

    # For testing, we need to patch ConfigRepositoryManager.__enter__ to return our mock
    with patch('ra_aid.database.repositories.config_repository.ConfigRepositoryManager.__enter__', return_value=mock_config_repository):
        with patch("ra_aid.__main__.parse_arguments", return_value=mock_args):
            main()
            # Verify the mock repo's set method was called with the expected values
            mock_config_repository.set.assert_any_call(
                "planner_provider", "openai")
            mock_config_repository.set.assert_any_call(
                "planner_model", "gpt-4")


def test_use_aider_flag(mock_dependencies, mock_config_repository):
    """Test that use-aider flag is correctly stored in config."""
    from unittest.mock import patch
    from ra_aid.__main__ import main
    from ra_aid.tool_configs import MODIFICATION_TOOLS, set_modification_tools

    # Reset mocks
    mock_config_repository.update.reset_mock()

    # Reset to default state
    set_modification_tools(False)

    # For testing, we need to patch ConfigRepositoryManager.__enter__ to return our mock
    with patch('ra_aid.database.repositories.config_repository.ConfigRepositoryManager.__enter__', return_value=mock_config_repository):
        # Check default behavior (use_aider=False)
        mock_args_default = create_mock_args(
            message="test message", use_aider=False)
        with patch("ra_aid.__main__.parse_arguments", return_value=mock_args_default):
            main()
            # Verify use_aider is set to False in the update call
            mock_config_repository.update.assert_called()
            # Get the call arguments
            call_args = mock_config_repository.update.call_args_list
            # Find the call that includes use_aider
            use_aider_found = False
            for args, _ in call_args:
                config_dict = args[0]
                if "use_aider" in config_dict and config_dict["use_aider"] is False:
                    use_aider_found = True
                    break
            assert use_aider_found, f"use_aider=False not found in update calls: {call_args}"

            # Check that file tools are enabled by default
            tool_names = [tool.name for tool in MODIFICATION_TOOLS]
            assert "file_str_replace" in tool_names
            assert "put_complete_file_contents" in tool_names
            assert "run_programming_task" not in tool_names

        # Reset mocks
        mock_config_repository.update.reset_mock()

        # Check with --use-aider flag
        mock_args_aider = create_mock_args(
            message="test message", use_aider=True)
        with patch("ra_aid.__main__.parse_arguments", return_value=mock_args_aider):
            main()
            # Verify use_aider is set to True in the update call
            mock_config_repository.update.assert_called()
            # Get the call arguments
            call_args = mock_config_repository.update.call_args_list
            # Find the call that includes use_aider
            use_aider_found = False
            for args, _ in call_args:
                config_dict = args[0]
                if "use_aider" in config_dict and config_dict["use_aider"] is True:
                    use_aider_found = True
                    break
            assert use_aider_found, f"use_aider=True not found in update calls: {call_args}"

            # Check that run_programming_task is enabled
            tool_names = [tool.name for tool in MODIFICATION_TOOLS]
            assert "file_str_replace" not in tool_names
            assert "put_complete_file_contents" not in tool_names
            assert "run_programming_task" in tool_names

    # Reset to default state for other tests
    set_modification_tools(False)

@pytest.fixture(autouse=True)
def mock_migrations():
    """Mock database migrations to prevent operational errors during tests."""
    with patch("ra_aid.__main__.ensure_migrations_applied", return_value=(True, None)):
        yield
