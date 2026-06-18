"""
Two-Layer Blockchain Audit Bridge

Compiles both Solidity contracts with py-solc-x, deploys them to an
in-process eth-tester EVM (local node) and exposes:

    AuditWriter.write_audit_record()        → Layer 1: immutable log
    AuditWriter.write_governance_decision() → Layer 2: mutable verdict
    AuditWriter.reconstruct_run()           → rebuild decision from chain
    AuditWriter.reconstruct_all()           → audit completeness sweep

Factory function:
    writer = AuditWriter.create_local()
    # deploys fresh contracts to an in-process EVM
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

CONTRACTS_DIR    = Path(__file__).parent / "contracts"
PIPELINE_VERSION = 4       # C4 = full bio-inspired pipeline
_SOLC_VERSION    = "0.8.28"


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class WriteReceipt:
    run_id:        str
    layer:         str      # "audit" | "governance"
    tx_hash:       str
    block_number:  int
    gas_used:      int
    write_time_ms: float


@dataclass
class ReconstructedRun:
    run_id:             str
    complete:           bool
    audit_layer:        Optional[dict] = None
    governance_layer:   Optional[dict] = None
    missing_layers:     list = field(default_factory=list)


# ── contract compilation ──────────────────────────────────────────────────────

def _compile_contracts() -> dict:
    """Compile AuditLog.sol and GovernanceDecision.sol; return {name: {abi, bin}}."""
    import solcx

    installed = [str(v) for v in solcx.get_installed_solc_versions()]
    if _SOLC_VERSION not in installed:
        print(f"  [audit_writer] Installing solc {_SOLC_VERSION}…")
        solcx.install_solc(_SOLC_VERSION)
    solcx.set_solc_version(_SOLC_VERSION)

    # compile each contract individually to avoid duplicate SPDX header errors
    audit_compiled = solcx.compile_source(
        (CONTRACTS_DIR / "AuditLog.sol").read_text(),
        output_values=["abi", "bin"],
        solc_version=_SOLC_VERSION,
    )
    gov_compiled = solcx.compile_source(
        (CONTRACTS_DIR / "GovernanceDecision.sol").read_text(),
        output_values=["abi", "bin"],
        solc_version=_SOLC_VERSION,
    )
    return {
        "AuditLog":           audit_compiled["<stdin>:AuditLog"],
        "GovernanceDecision": gov_compiled["<stdin>:GovernanceDecision"],
    }


# ── helper: hashing ───────────────────────────────────────────────────────────

def _run_id_bytes(qid: str) -> bytes:
    """Derive a stable 32-byte run ID from a question ID string."""
    return hashlib.sha256(qid.encode()).digest()


def _text_hash(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


# ── main class ────────────────────────────────────────────────────────────────

class AuditWriter:
    """Python bridge to the two-layer blockchain audit system."""

    def __init__(self, w3, audit_contract, governance_contract):
        self._w3         = w3
        self._account    = w3.eth.accounts[0]
        self._audit      = audit_contract
        self._governance = governance_contract

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create_local(cls) -> "AuditWriter":
        """
        Spin up an in-process EVM (eth-tester / py-evm) and deploy both
        contracts. Returns a ready-to-use AuditWriter.
        """
        from web3 import Web3
        from eth_tester import EthereumTester, PyEVMBackend

        print("  [audit_writer] Starting in-process EVM (py-evm)…")
        tester = EthereumTester(PyEVMBackend())
        w3     = Web3(Web3.EthereumTesterProvider(tester))
        print(f"  [audit_writer] Connected: block={w3.eth.block_number}  "
              f"account={w3.eth.accounts[0][:10]}…")

        artifacts = _compile_contracts()
        account   = w3.eth.accounts[0]

        # deploy AuditLog (Layer 1)
        AuditLog = w3.eth.contract(
            abi=artifacts["AuditLog"]["abi"],
            bytecode=artifacts["AuditLog"]["bin"],
        )
        tx   = AuditLog.constructor().transact({"from": account})
        rec  = w3.eth.wait_for_transaction_receipt(tx)
        audit_contract = w3.eth.contract(
            address=rec["contractAddress"],
            abi=artifacts["AuditLog"]["abi"],
        )
        print(f"  [audit_writer] AuditLog deployed at {rec['contractAddress']}")

        # deploy GovernanceDecision (Layer 2)
        GovDec = w3.eth.contract(
            abi=artifacts["GovernanceDecision"]["abi"],
            bytecode=artifacts["GovernanceDecision"]["bin"],
        )
        tx  = GovDec.constructor().transact({"from": account})
        rec = w3.eth.wait_for_transaction_receipt(tx)
        governance_contract = w3.eth.contract(
            address=rec["contractAddress"],
            abi=artifacts["GovernanceDecision"]["abi"],
        )
        print(f"  [audit_writer] GovernanceDecision deployed at {rec['contractAddress']}")

        return cls(w3, audit_contract, governance_contract)

    # ── write operations ──────────────────────────────────────────────────────

    def write_audit_record(
        self, qid: str, question: str, answer: str
    ) -> WriteReceipt:
        """Write an immutable audit record to Layer 1."""
        run_id  = _run_id_bytes(qid)
        q_hash  = _text_hash(question)
        a_hash  = _text_hash(answer)

        t0 = time.perf_counter()
        tx_hash = self._audit.functions.logRun(
            run_id, q_hash, a_hash, PIPELINE_VERSION
        ).transact({"from": self._account})
        receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash)
        write_ms = (time.perf_counter() - t0) * 1000

        return WriteReceipt(
            run_id=qid,
            layer="audit",
            tx_hash=tx_hash.hex(),
            block_number=receipt["blockNumber"],
            gas_used=receipt["gasUsed"],
            write_time_ms=round(write_ms, 3),
        )

    def write_governance_decision(
        self,
        qid:                str,
        flagged:            bool,
        confidence_score:   float,
        contradiction_score: float,
        verdict_reason:     str = "",
    ) -> WriteReceipt:
        """Write a governance decision to Layer 2."""
        run_id        = _run_id_bytes(qid)
        conf_scaled   = int(confidence_score   * 10_000)
        contra_scaled = int(contradiction_score * 10_000)

        t0 = time.perf_counter()
        tx_hash = self._governance.functions.recordDecision(
            run_id, flagged, conf_scaled, contra_scaled, verdict_reason[:200]
        ).transact({"from": self._account})
        receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash)
        write_ms = (time.perf_counter() - t0) * 1000

        return WriteReceipt(
            run_id=qid,
            layer="governance",
            tx_hash=tx_hash.hex(),
            block_number=receipt["blockNumber"],
            gas_used=receipt["gasUsed"],
            write_time_ms=round(write_ms, 3),
        )

    # ── read / reconstruct ────────────────────────────────────────────────────

    def reconstruct_run(self, qid: str) -> ReconstructedRun:
        """
        Attempt to reconstruct a pipeline run from chain records alone.
        Returns a ReconstructedRun indicating which layers were found.
        """
        run_id  = _run_id_bytes(qid)
        missing = []

        audit_data = None
        try:
            e = self._audit.functions.getEntry(run_id).call()
            audit_data = {
                "run_id":           e[0].hex(),
                "timestamp":        e[1],
                "question_hash":    e[2].hex(),
                "answer_hash":      e[3].hex(),
                "pipeline_version": e[4],
            }
        except Exception:
            missing.append("audit")

        gov_data = None
        try:
            d = self._governance.functions.getDecision(run_id).call()
            gov_data = {
                "flagged":             d[1],
                "confidence_score":    d[2] / 10_000,
                "contradiction_score": d[3] / 10_000,
                "verdict_reason":      d[4],
                "overridden":          d[5],
                "override_reason":     d[6],
                "recorded_at":         d[7],
            }
        except Exception:
            missing.append("governance")

        return ReconstructedRun(
            run_id=qid,
            complete=(len(missing) == 0),
            audit_layer=audit_data,
            governance_layer=gov_data,
            missing_layers=missing,
        )

    def reconstruct_all(self, qids: list) -> list:
        """Reconstruct all runs; returns list of ReconstructedRun."""
        return [self.reconstruct_run(qid) for qid in qids]

    # ── stats ─────────────────────────────────────────────────────────────────

    def total_audit_entries(self) -> int:
        return self._audit.functions.totalEntries().call()

    def total_governance_decisions(self) -> int:
        return self._governance.functions.totalDecisions().call()

    def audit_address(self) -> str:
        return self._audit.address

    def governance_address(self) -> str:
        return self._governance.address
