#!/usr/bin/env python3
"""
PHASE 1 VALIDATION SCRIPT
Automatically checks if the microservice architecture is ready to test.

Usage:
    python validate_phase_1.py

Status Codes:
    0 = All checks passed, ready to test
    1 = Some checks failed, see report
    2 = Critical issues found
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# ANSI Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class ValidationReport:
    def __init__(self):
        self.checks = []
        self.warnings = []
        self.errors = []
        self.critical = []
    
    def add_check(self, name: str, passed: bool, message: str = ""):
        self.checks.append({
            'name': name,
            'passed': passed,
            'message': message
        })
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def add_error(self, message: str):
        self.errors.append(message)
    
    def add_critical(self, message: str):
        self.critical.append(message)
    
    def print_report(self):
        """Print formatted validation report"""
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}PHASE 1 VALIDATION REPORT{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        # Summary
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c['passed'])
        failed = total - passed
        
        print(f"{BOLD}Summary:{RESET}")
        print(f"  Total Checks: {total}")
        print(f"  {GREEN}✓ Passed: {passed}{RESET}")
        print(f"  {RED}✗ Failed: {failed}{RESET}")
        print(f"  {YELLOW}⚠ Warnings: {len(self.warnings)}{RESET}")
        print()
        
        # Detailed checks
        print(f"{BOLD}Detailed Results:{RESET}")
        for check in self.checks:
            status = f"{GREEN}✓{RESET}" if check['passed'] else f"{RED}✗{RESET}"
            print(f"  {status} {check['name']}")
            if check['message']:
                print(f"      {check['message']}")
        
        # Warnings
        if self.warnings:
            print(f"\n{YELLOW}{BOLD}Warnings:{RESET}")
            for warning in self.warnings:
                print(f"  {YELLOW}⚠{RESET} {warning}")
        
        # Errors
        if self.errors:
            print(f"\n{RED}{BOLD}Errors:{RESET}")
            for error in self.errors:
                print(f"  {RED}✗{RESET} {error}")
        
        # Critical
        if self.critical:
            print(f"\n{RED}{BOLD}CRITICAL ISSUES:{RESET}")
            for issue in self.critical:
                print(f"  {RED}🔴{RESET} {issue}")
        
        # Final status
        print(f"\n{BOLD}{'='*70}{RESET}")
        if self.critical:
            print(f"{RED}STATUS: CRITICAL ISSUES - CANNOT TEST{RESET}")
            print(f"{RED}Fix critical issues before proceeding{RESET}")
            status = 2
        elif self.errors or failed > 0:
            print(f"{YELLOW}STATUS: ISSUES FOUND - MAY NOT WORK{RESET}")
            print(f"{YELLOW}Review and fix errors before testing{RESET}")
            status = 1
        else:
            print(f"{GREEN}STATUS: ALL CHECKS PASSED - READY TO TEST!{RESET}")
            print(f"{GREEN}Run: powershell -File start_services.ps1{RESET}")
            status = 0
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        return status


def check_python_version() -> Tuple[bool, str]:
    """Check if Python 3.11+ is installed"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 11:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor} (need 3.11+)"


def check_file_exists(path: str, description: str = "") -> Tuple[bool, str]:
    """Check if a file/folder exists"""
    exists = Path(path).exists()
    desc = description or path
    return exists, f"{desc}: {path}"


def check_python_package(package: str) -> Tuple[bool, str]:
    """Check if a Python package is installed"""
    try:
        __import__(package)
        return True, f"{package} installed"
    except ImportError:
        return False, f"{package} NOT installed"


def check_port_available(port: int) -> Tuple[bool, str]:
    """Check if a port is available (simple check)"""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        if result != 0:
            return True, f"Port {port} is available"
        return False, f"Port {port} is in use (something is already running)"
    except Exception as e:
        return False, f"Cannot check port {port}: {e}"


def check_venv_exists() -> Tuple[bool, str]:
    """Check if virtual environment exists"""
    venv_path = Path("ml-backend/venv")
    if venv_path.exists():
        return True, "Virtual environment exists in ml-backend/venv"
    return False, "Virtual environment NOT found at ml-backend/venv"


def check_requirements_updated() -> Tuple[bool, str]:
    """Check if requirements.txt has been updated with new packages"""
    req_file = Path("ml-backend/requirements.txt")
    if not req_file.exists():
        return False, "requirements.txt not found"
    
    content = req_file.read_text()
    
    required = ['fastapi', 'uvicorn', 'streamlit', 'requests', 'pytest']
    found = [pkg for pkg in required if pkg.lower() in content.lower()]
    
    if len(found) == len(required):
        return True, f"All packages found ({', '.join(found)})"
    
    missing = [pkg for pkg in required if pkg not in found]
    return False, f"Missing packages: {', '.join(missing)}"


def check_git_history() -> Tuple[bool, str]:
    """Check if git repository is initialized"""
    if Path(".git").exists():
        return True, "Git repository initialized with history"
    return False, "Git repository not found"


def validate_phase_1() -> int:
    """Run all validation checks"""
    report = ValidationReport()
    
    print(f"{BLUE}{BOLD}Starting Phase 1 Validation...{RESET}\n")
    
    # ==================== CRITICAL CHECKS ====================
    print(f"{BOLD}[1/6] Checking Critical Requirements...{RESET}")
    
    # Python version
    passed, msg = check_python_version()
    report.add_check("Python Version", passed, msg)
    if not passed:
        report.add_critical(f"Python 3.11+ required. {msg}")
    
    # Venv
    passed, msg = check_venv_exists()
    report.add_check("Virtual Environment", passed, msg)
    if not passed:
        report.add_critical("Virtual environment not found. Run setup-backend.sh first.")
    
    # ==================== STRUCTURE CHECKS ====================
    print(f"{BOLD}[2/6] Checking Directory Structure...{RESET}")
    
    structure_checks = [
        ("ml-backend/", "Backend directory"),
        ("ml-backend/app.py", "FastAPI application"),
        ("ml-backend/src/ds_covid_backend/", "DDD structure"),
        ("ml-backend/src/ds_covid_backend/api/", "API layer"),
        ("ml-backend/src/ds_covid_backend/domain/", "Domain layer"),
        ("ml-backend/src/ds_covid_backend/application/", "Application layer"),
        ("ml-backend/src/ds_covid_backend/infrastructure/", "Infrastructure layer"),
        ("ml-backend/src/ds_covid_backend/config/", "Config layer"),
        ("ml-backend/tests/", "Tests directory"),
        ("streamlit_app.py", "Streamlit frontend"),
        ("_REFACTORING_MICROSERVICE_/", "Documentation"),
        ("migration_backup/", "Backup directory"),
        ("_OLD_ROOT_FILES/", "Archive directory"),
    ]
    
    for path, desc in structure_checks:
        passed, msg = check_file_exists(path, desc)
        report.add_check(f"Exists: {desc}", passed)
    
    # ==================== ROOT CLEANUP CHECKS ====================
    print(f"{BOLD}[3/6] Checking Root Directory Cleanup...{RESET}")
    
    # Check that setup.py is NOT in root (should be archived)
    root_setup_exists = Path("setup.py").exists()
    report.add_check("setup.py Archived", not root_setup_exists)
    if root_setup_exists:
        report.add_error("setup.py still in root - should be archived to _OLD_ROOT_FILES/")
    
    # Check that old src/ is NOT in root
    root_src_exists = Path("src").exists()
    report.add_check("src/ Archived", not root_src_exists)
    if root_src_exists:
        report.add_error("src/ still in root - should be archived to _OLD_ROOT_FILES/")
    
    # ==================== SCRIPTS CHECKS ====================
    print(f"{BOLD}[4/6] Checking Launch & Test Scripts...{RESET}")
    
    scripts = [
        "start_services.ps1",
        "start_services.sh",
        "test_microservices.ps1",
        "test_microservices.sh",
    ]
    
    for script in scripts:
        passed, msg = check_file_exists(script, f"Script: {script}")
        report.add_check(f"Script exists: {script}", passed)
    
    # ==================== DEPENDENCIES CHECKS ====================
    print(f"{BOLD}[5/6] Checking Python Dependencies...{RESET}")
    
    # Check requirements.txt
    passed, msg = check_requirements_updated()
    report.add_check("Requirements.txt Updated", passed, msg)
    
    # Check key packages
    packages = ['fastapi', 'streamlit', 'pytest', 'pandas', 'numpy']
    for pkg in packages:
        passed, msg = check_python_package(pkg)
        report.add_check(f"Package: {pkg}", passed, msg)
        if not passed:
            report.add_warning(f"{pkg} not installed - will be installed on first run")
    
    # ==================== PORT CHECKS ====================
    print(f"{BOLD}[6/6] Checking Port Availability...{RESET}")
    
    ports = [
        (8000, "Backend (FastAPI)"),
        (8501, "Frontend (Streamlit)"),
    ]
    
    for port, service in ports:
        passed, msg = check_port_available(port)
        report.add_check(f"Port {port} Available ({service})", passed, msg)
        if not passed:
            report.add_error(f"Port {port} is in use. Kill the process using it.")
    
    # ==================== GIT CHECKS ====================
    print(f"{BOLD}[Bonus] Checking Git History...{RESET}")
    
    passed, msg = check_git_history()
    report.add_check("Git Repository", passed, msg)
    
    # Print report and return status
    return report.print_report()


def main():
    """Main entry point"""
    try:
        status = validate_phase_1()
        sys.exit(status)
    except KeyboardInterrupt:
        print(f"\n{RED}Validation interrupted by user{RESET}")
        sys.exit(2)
    except Exception as e:
        print(f"\n{RED}Validation error: {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
