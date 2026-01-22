"""Test cases for legal document analysis."""
from typing import List, Dict, Any


LEGAL_TEST_CASES = [
    {
        'id': 'test_001',
        'question': 'What is the defect liability period?',
        'expected_keywords': ['defect liability', 'period', 'years'],
        'category': 'temporal',
        'expected_answer_type': 'specific_period'
    },
    {
        'id': 'test_002',
        'question': 'What are the payment terms?',
        'expected_keywords': ['payment', 'terms', 'amount'],
        'category': 'financial',
        'expected_answer_type': 'terms'
    },
    {
        'id': 'test_003',
        'question': 'What happens if the contractor defaults?',
        'expected_keywords': ['default', 'termination', 'penalty'],
        'category': 'termination',
        'expected_answer_type': 'consequence'
    },
    {
        'id': 'test_004',
        'question': 'What are the performance security requirements?',
        'expected_keywords': ['performance', 'security', 'guarantee', 'bank'],
        'category': 'security',
        'expected_answer_type': 'requirement'
    },
    {
        'id': 'test_005',
        'question': 'What are the key obligations of the contractor?',
        'expected_keywords': ['obligation', 'contractor', 'commitment'],
        'category': 'obligations',
        'expected_answer_type': 'list'
    },
    {
        'id': 'test_006',
        'question': 'What is the project description?',
        'expected_keywords': ['construction', 'four lane', 'highway'],
        'category': 'general',
        'expected_answer_type': 'description'
    },
    {
        'id': 'test_007',
        'question': 'What are the disqualification criteria?',
        'expected_keywords': ['disqualification', 'exclusion', 'violation'],
        'category': 'compliance',
        'expected_answer_type': 'criteria'
    },
    {
        'id': 'test_008',
        'question': 'What compensation is provided for damages?',
        'expected_keywords': ['compensation', 'damage', 'liquidated'],
        'category': 'compensation',
        'expected_answer_type': 'amount_or_terms'
    }
]


def get_test_cases() -> List[Dict[str, Any]]:
    """Get all test cases."""
    return LEGAL_TEST_CASES


def get_test_case_by_id(test_id: str) -> Dict[str, Any]:
    """Get test case by ID."""
    for test_case in LEGAL_TEST_CASES:
        if test_case['id'] == test_id:
            return test_case
    return None


def get_test_cases_by_category(category: str) -> List[Dict[str, Any]]:
    """Get test cases by category."""
    return [tc for tc in LEGAL_TEST_CASES if tc.get('category') == category]
