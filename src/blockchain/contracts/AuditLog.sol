// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/**
 * Layer 1 — Immutable Audit Log
 *
 * Every pipeline run is permanently recorded here. Records cannot be
 * modified or deleted after creation. Provides the tamper-evident
 * foundation for the two-layer blockchain governance architecture.
 *
 * Storage layout per entry:
 *   runId           — keccak256(qid) truncated to 32 bytes
 *   timestamp       — block.timestamp at submission
 *   questionHash    — sha256(question_text)
 *   answerHash      — sha256(answer_text)
 *   pipelineVersion — 4 = C4 full pipeline
 */
contract AuditLog {

    struct LogEntry {
        bytes32 runId;
        uint256 timestamp;
        bytes32 questionHash;
        bytes32 answerHash;
        uint8   pipelineVersion;
    }

    LogEntry[] private _entries;
    mapping(bytes32 => uint256) private _runIndex;   // runId → 1-based index

    event RunLogged(
        bytes32 indexed runId,
        uint256 timestamp,
        bytes32 questionHash,
        bytes32 answerHash,
        uint8   pipelineVersion
    );

    function logRun(
        bytes32 runId,
        bytes32 questionHash,
        bytes32 answerHash,
        uint8   pipelineVersion
    ) external {
        require(_runIndex[runId] == 0, "AuditLog: duplicate runId");
        _entries.push(LogEntry({
            runId:           runId,
            timestamp:       block.timestamp,
            questionHash:    questionHash,
            answerHash:      answerHash,
            pipelineVersion: pipelineVersion
        }));
        _runIndex[runId] = _entries.length;
        emit RunLogged(runId, block.timestamp, questionHash, answerHash, pipelineVersion);
    }

    function getEntry(bytes32 runId) external view returns (LogEntry memory) {
        uint256 idx = _runIndex[runId];
        require(idx != 0, "AuditLog: runId not found");
        return _entries[idx - 1];
    }

    function getEntryByIndex(uint256 idx) external view returns (LogEntry memory) {
        require(idx < _entries.length, "AuditLog: index out of bounds");
        return _entries[idx];
    }

    function totalEntries() external view returns (uint256) {
        return _entries.length;
    }

    function exists(bytes32 runId) external view returns (bool) {
        return _runIndex[runId] != 0;
    }
}
