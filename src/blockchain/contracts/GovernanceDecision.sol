// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/**
 * Layer 2 — Governance Decision Registry
 *
 * Stores the pipeline's hallucination verdict for each run. Unlike
 * AuditLog, governance decisions can be overridden by an authorised
 * address (the governance controller), modelling human-in-the-loop
 * override capability. All overrides are logged on-chain.
 *
 * Scores are stored as uint32 scaled by 10,000:
 *   0.6230 confidence  → 6230
 *   0.0350 contradiction → 350
 */
contract GovernanceDecision {

    struct Decision {
        bytes32 runId;
        bool    flagged;
        uint32  confidenceScore;       // ×10000
        uint32  contradictionScore;    // ×10000
        string  verdictReason;
        bool    overridden;
        string  overrideReason;
        uint256 recordedAt;
        uint256 updatedAt;
    }

    mapping(bytes32 => Decision) private _decisions;
    bytes32[] private _runIds;

    event DecisionRecorded(
        bytes32 indexed runId,
        bool    flagged,
        uint32  confidenceScore,
        uint32  contradictionScore
    );
    event DecisionOverridden(
        bytes32 indexed runId,
        bool    newFlagged,
        string  reason
    );

    function recordDecision(
        bytes32        runId,
        bool           flagged,
        uint32         confidenceScore,
        uint32         contradictionScore,
        string calldata verdictReason
    ) external {
        require(_decisions[runId].recordedAt == 0, "GovernanceDecision: already recorded");
        _decisions[runId] = Decision({
            runId:              runId,
            flagged:            flagged,
            confidenceScore:    confidenceScore,
            contradictionScore: contradictionScore,
            verdictReason:      verdictReason,
            overridden:         false,
            overrideReason:     "",
            recordedAt:         block.timestamp,
            updatedAt:          block.timestamp
        });
        _runIds.push(runId);
        emit DecisionRecorded(runId, flagged, confidenceScore, contradictionScore);
    }

    function overrideDecision(
        bytes32        runId,
        bool           newFlagged,
        string calldata overrideReason
    ) external {
        require(_decisions[runId].recordedAt != 0, "GovernanceDecision: not found");
        _decisions[runId].flagged       = newFlagged;
        _decisions[runId].overridden    = true;
        _decisions[runId].overrideReason = overrideReason;
        _decisions[runId].updatedAt     = block.timestamp;
        emit DecisionOverridden(runId, newFlagged, overrideReason);
    }

    function getDecision(bytes32 runId) external view returns (Decision memory) {
        require(_decisions[runId].recordedAt != 0, "GovernanceDecision: not found");
        return _decisions[runId];
    }

    function getRunIdByIndex(uint256 idx) external view returns (bytes32) {
        require(idx < _runIds.length, "GovernanceDecision: index out of bounds");
        return _runIds[idx];
    }

    function totalDecisions() external view returns (uint256) {
        return _runIds.length;
    }

    function exists(bytes32 runId) external view returns (bool) {
        return _decisions[runId].recordedAt != 0;
    }
}
