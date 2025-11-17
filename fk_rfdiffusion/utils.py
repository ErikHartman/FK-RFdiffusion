from pathlib import Path
from Bio.PDB import MMCIFParser, PDBIO
import tempfile
import os
import re
import random
from typing import List


def infer_design_mode_from_contigs(contigs: List[str]) -> str:
    """Infer design mode from contig specifications - binder if contains chain identifiers, unconditional otherwise."""
    for contig in contigs:
        if any(c == '/' for c in contig):
            return "binder"
    return "unconditional"


def get_checkpoint_path(design_mode: str, checkpoint: str) -> str:
    """Get checkpoint path based on design mode and checkpoint type"""
    pepdiff_root = Path(__file__).parent.parent
    
    if design_mode == "unconditional":
        # Force use of Base checkpoint for unconditional design
        return str(pepdiff_root / "externals/RFdiffusion/models/Base_ckpt.pt")
    elif design_mode == "binder":
        if checkpoint.lower() == "beta":
            return str(pepdiff_root / "externals/RFdiffusion/models/Complex_beta_ckpt.pt")
        else:
            return str(pepdiff_root / "externals/RFdiffusion/models/Complex_base_ckpt.pt")
    else:
        raise ValueError(f"Invalid design_mode: {design_mode}. Must be 'binder' or 'unconditional'")

def parse_chain_assignments(contigs):
    """
    Parse contig specifications to automatically determine target and design chains.
    """
    target_chains = set()
    for contig in contigs:
        segments = contig.replace('/', ' ').split()
        
        for segment in segments:
            if segment and segment[0].isalpha():
                chain_letter = segment[0].upper()
                target_chains.add(chain_letter)
    
    
    all_chains = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    available_chains = all_chains - target_chains
    design_chain = min(available_chains) 
    target_chain = min(target_chains)
    
    return target_chain, design_chain


def parse_structure_file(file_path: str) -> str:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Structure file not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == '.pdb':
        return str(file_path)
    elif suffix in ['.cif', '.mmcif']:
        return _convert_mmcif_to_pdb(str(file_path))
    
    else:
        raise ValueError(f"Unsupported structure file format: {suffix}. "
                        f"Supported formats: .pdb, .cif, .mmcif")


def _convert_mmcif_to_pdb(mmcif_path: str) -> str:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", mmcif_path)
    
    temp_fd, temp_pdb_path = tempfile.mkstemp(suffix='.pdb', prefix='converted_')
    os.close(temp_fd)  
    
    io = PDBIO()
    io.set_structure(structure)
    io.save(temp_pdb_path)
    
    return temp_pdb_path


def cleanup_temp_pdb(pdb_path: str, original_path: str) -> None:
    original_suffix = Path(original_path).suffix.lower()
    if original_suffix in ['.cif', '.mmcif'] and pdb_path != original_path:
        try:
            os.unlink(pdb_path)
        except (OSError, FileNotFoundError):
            pass


def detect_length_ranges(contigs: List[str]) -> bool:
    """Detect if any contig contains length ranges (e.g., "50-75")."""
    range_pattern = r'\b(\d+)-(\d+)\b'
    for contig in contigs:
        if re.search(range_pattern, contig):
            return True
    return False


def sample_contig_lengths(contigs: List[str], n_runs: int) -> List[List[str]]:
    """Sample specific lengths from contig ranges for multiple runs."""
    range_pattern = r'\b(\d+)-(\d+)\b'
    sampled_contigs = []
    
    for run_id in range(n_runs):
        run_contigs = []
        for contig in contigs:
            modified_contig = contig
            matches = list(re.finditer(range_pattern, contig))
            
            # Process matches in reverse order to avoid index shifts
            for match in reversed(matches):
                start_pos, end_pos = match.span()
                min_len = int(match.group(1))
                max_len = int(match.group(2))
                sampled_length = random.randint(min_len, max_len)
                modified_contig = (modified_contig[:start_pos] + 
                                 str(sampled_length) + 
                                 modified_contig[end_pos:])
            run_contigs.append(modified_contig)
        sampled_contigs.append(run_contigs)
    
    return sampled_contigs


def get_symmetry_order(symmetry: str) -> int:
    """
    Extract the numerical order from a symmetry string.
    """
    if symmetry is None:
        return 1
    
    symmetry_lower = symmetry.lower()
    if symmetry_lower.startswith('c'):
        if symmetry[1:].isdigit():
            return int(symmetry[1:])
        raise ValueError(f"Invalid cyclic symmetry format: {symmetry}")
    elif symmetry_lower.startswith('d'):
        if symmetry[1:].isdigit():
            return int(symmetry[1:]) * 2  # Dihedral has 2n subunits
        raise ValueError(f"Invalid dihedral symmetry format: {symmetry}")
    elif symmetry_lower in ['t3']:
        return 4

    else:
        raise ValueError(f"Unrecognized symmetry: {symmetry}")


def validate_symmetry_contigs(contigs: List[str], symmetry: str) -> None:
    """
    Validate that contig lengths are compatible with the specified symmetry.
    
    For symmetric assemblies, the total length must be divisible by the symmetry order.
    """
    if symmetry is None:
        return
    
    order = get_symmetry_order(symmetry)
    
    # Parse total length from contigs
    for contig in contigs:
        # For unconditional design, contigs are just numbers or ranges
        # Extract all numbers (ignoring chain identifiers and slashes)
        numbers = re.findall(r'\b(\d+)\b', contig.replace('-', ' '))
        
        if numbers:
            # For symmetry, we expect a single total length specification
            total_length = int(numbers[0])
            
            if total_length % order != 0:
                raise ValueError(
                    f"Contig length {total_length} is not divisible by symmetry order {order} "
                    f"for {symmetry} symmetry. Each subunit would have {total_length}/{order} = "
                    f"{total_length/order:.2f} residues. Please use a length divisible by {order}."
                )