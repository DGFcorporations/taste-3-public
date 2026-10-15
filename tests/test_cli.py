import importlib
import json

import pytest

from taste.models import Standard
from taste.store import Store


def cli():
    try:
        return importlib.import_module("taste.cli")
    except ModuleNotFoundError:
        pytest.fail("Missing owner CLI implementation")


def test_owner_decline_never_activates_and_confirmation_binds_current_revision(
    monkeypatch, tmp_path, capsys
):
    path = tmp_path / "owner.db"
    with Store(path) as store:
        store.propose(
            Standard(
                id="proposal",
                profile_id="p",
                category="structure",
                polarity="do",
                directive="Synthetic exact directive",
                rationale="Synthetic original rationale",
            )
        )
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    assert (
        cli().main(
            [
                "--db",
                str(path),
                "approve",
                "proposal",
                "--profile",
                "p",
                "--rationale",
                "Synthetic approval reason",
            ]
        )
        == 1
    )
    displayed = capsys.readouterr().out
    assert "Synthetic exact directive" in displayed
    assert "Polarity: do" in displayed
    assert "Category: structure" in displayed
    with Store(path) as store:
        assert not store.snapshot("p").standards[0]["_local_approved"]
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    assert (
        cli().main(
            [
                "--db",
                str(path),
                "approve",
                "proposal",
                "--profile",
                "p",
                "--rationale",
                "Synthetic approval reason",
            ]
        )
        == 0
    )
    with Store(path) as store:
        assert store.snapshot("p").standards[0]["_local_approved"]


def test_cli_export_filter_and_sanitized_missing_record(monkeypatch, tmp_path, capsys):
    path = tmp_path / "cli.db"
    with Store(path) as store:
        store.propose(
            Standard(
                id="s", profile_id="p", category="structure", polarity="do", directive="Synthetic"
            )
        )
    assert cli().main(["--db", str(path), "export", "--profile", "p", "--state", "retired"]) == 0
    assert json.loads(capsys.readouterr().out) == {"standards": []}
    assert (
        cli().main(
            ["--db", str(path), "approve", "missing", "--profile", "p", "--rationale", "Synthetic"]
        )
        == 1
    )
    assert "error" in capsys.readouterr().err


def test_cli_rejects_automatic_confirmation_flag(tmp_path):
    with pytest.raises(SystemExit):
        cli().main(
            [
                "--db",
                str(tmp_path / "never.db"),
                "approve",
                "s",
                "--rationale",
                "Synthetic",
                "--yes",
            ]
        )


def test_cli_import_supersession_history_and_drift(monkeypatch, tmp_path, capsys):
    path = tmp_path / "lifecycle.db"
    source = tmp_path / "portable.json"
    source.write_text(
        json.dumps(
            {
                "standards": [
                    {
                        "id": "old",
                        "profile_id": "p",
                        "category": "structure",
                        "polarity": "do",
                        "directive": "Synthetic old lead",
                        "rationale": "Synthetic reason",
                        "state": "provisional",
                        "confidence": 0.5,
                        "provenance": "persona",
                        "supersedes": [],
                        "created": "2026-01-01T00:00:00Z",
                        "updated": "2026-01-01T00:00:00Z",
                    },
                    {
                        "id": "new",
                        "profile_id": "p",
                        "category": "structure",
                        "polarity": "do",
                        "directive": "Synthetic new lead",
                        "rationale": "Synthetic replacement",
                        "state": "provisional",
                        "confidence": 0.5,
                        "provenance": "persona",
                        "supersedes": [],
                        "created": "2026-01-01T00:00:00Z",
                        "updated": "2026-01-01T00:00:00Z",
                    },
                ]
            }
        )
    )
    assert cli().main(["--db", str(path), "import", str(source), "--profile", "p"]) == 0
    assert json.loads(capsys.readouterr().out)["imported"] == 2
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    assert (
        cli().main(
            [
                "--db",
                str(path),
                "approve",
                "old",
                "--profile",
                "p",
                "--rationale",
                "Synthetic owner reason",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        cli().main(
            [
                "--db",
                str(path),
                "supersede",
                "new",
                "old",
                "--profile",
                "p",
                "--rationale",
                "Synthetic better order",
            ]
        )
        == 0
    )
    review = capsys.readouterr().out
    assert "Synthetic old lead" in review and "Synthetic new lead" in review
    assert cli().main(["--db", str(path), "history", "old", "--profile", "p"]) == 0
    assert [row["id"] for row in json.loads(capsys.readouterr().out)["standards"]] == ["new", "old"]
    assert (
        cli().main(
            ["--db", str(path), "drift", "--profile", "p", "--as-of", "2100-01-01T00:00:00Z"]
        )
        == 0
    )
    assert [row["id"] for row in json.loads(capsys.readouterr().out)["stale"]] == ["old"]


def test_changed_revision_during_prompt_cannot_be_approved(monkeypatch, tmp_path, capsys):
    path = tmp_path / "stale.db"
    with Store(path) as store:
        store.propose(
            Standard(
                id="s",
                profile_id="p",
                category="structure",
                polarity="do",
                directive="Synthetic exact directive",
            )
        )

    def concurrent_approval(prompt):
        with Store(path) as store:
            standard = store.get_standard("p", "s")
            store.approve(
                "p",
                "s",
                "Synthetic concurrent reason",
                owner_confirmed=True,
                expected_updated=standard.updated,
            )
        return "yes"

    monkeypatch.setattr("builtins.input", concurrent_approval)
    assert (
        cli().main(
            [
                "--db",
                str(path),
                "approve",
                "s",
                "--profile",
                "p",
                "--rationale",
                "Synthetic stale reason",
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "state_conflict"
    with Store(path) as store:
        assert store.get_standard("p", "s").rationale == "Synthetic concurrent reason"


def test_export_replacement_failure_preserves_existing_file_and_cleans_temporary(
    monkeypatch, tmp_path, capsys
):
    output = tmp_path / "export.json"
    output.write_text("previous complete export")

    def interrupted_replace(source, destination):
        raise OSError("Synthetic interrupted replacement")

    monkeypatch.setattr("os.replace", interrupted_replace)
    assert cli().main(["export", "--output", str(output)]) == 1
    assert output.read_text() == "previous complete export"
    assert list(tmp_path.glob(".taste-export-*")) == []
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_request"


def test_export_atomically_replaces_existing_complete_file(tmp_path, capsys):
    output = tmp_path / "export.json"
    output.write_text("previous complete export")
    assert cli().main(["export", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == {"standards": []}
    assert list(tmp_path.glob(".taste-export-*")) == []
    assert capsys.readouterr().out == ""


def test_regular_approval_displays_declared_supersession_target(monkeypatch, tmp_path, capsys):
    path = tmp_path / "declared.db"
    with Store(path) as store:
        old = store.propose(Standard(id="old", profile_id="p", category="structure",
                                     polarity="do", directive="Synthetic old directive"))
        store.approve("p", "old", "Synthetic reason", owner_confirmed=True,
                      expected_updated=old.updated)
        store.propose(Standard(id="new", profile_id="p", category="structure",
                               polarity="do", directive="Synthetic new directive",
                               supersedes=["old"]))
    shown = []

    def confirm(prompt):
        shown.append(capsys.readouterr().out)
        return "yes"

    monkeypatch.setattr("builtins.input", confirm)
    assert cli().main(["--db", str(path), "approve", "new", "--profile", "p",
                       "--rationale", "Synthetic replacement reason"]) == 0
    assert "Replacement target: old" in shown[0]
    assert "Synthetic old directive" in shown[0]


def test_target_changed_during_prompt_rolls_back_supersession(monkeypatch, tmp_path, capsys):
    path = tmp_path / "target-race.db"
    with Store(path) as store:
        old = store.propose(Standard(id="old", profile_id="p", category="structure",
                                     polarity="do", directive="Synthetic old directive"))
        store.approve("p", "old", "Synthetic reason", owner_confirmed=True,
                      expected_updated=old.updated)
        store.propose(Standard(id="new", profile_id="p", category="structure",
                               polarity="do", directive="Synthetic new directive"))

    def concurrent_change(prompt):
        with Store(path) as store:
            old = store.get_standard("p", "old")
            store.approve("p", "old", "Synthetic concurrent revision", owner_confirmed=True,
                          expected_updated=old.updated)
        return "yes"

    monkeypatch.setattr("builtins.input", concurrent_change)
    assert cli().main(["--db", str(path), "supersede", "new", "old", "--profile", "p",
                       "--rationale", "Synthetic replacement reason"]) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "state_conflict"
    with Store(path) as store:
        assert store.get_standard("p", "old").state.value == "active"
        assert store.get_standard("p", "new").state.value == "provisional"
        assert store.snapshot("p").edges == []
