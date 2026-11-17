import pytest
from rl_retest.analysis import _paired
from scripts.analyze_study import estimate, paired


def row(task,method,value):
    return {'task_id':task,'method':method,'memory':'correct','expected_utility':value,'parameters':{'k':3,'horizon':2,'n':3,'p_c_current':.5,'cost':.02}}


def test_constant_paired_interval_and_missing_pairs():
    a=[row(str(i),'bounded',.8) for i in range(20)]
    b=[row(str(i),'iterative',.7) for i in range(20)]
    result=paired(a,b)
    assert result['mean']==pytest.approx(.1)
    assert result['ci_low']==pytest.approx(.1)
    assert result['ci_high']==pytest.approx(.1)
    with pytest.raises(ValueError):paired(a,b[:-1])
    with pytest.raises(ValueError):paired(a+a,b)


def test_setting_equal_weight_and_duplicate_budget_rejected():
    result=estimate({'small':[0],'large':[1]*20})
    assert result['mean']==pytest.approx(.5)
    a=row('x','bounded',.8);b=row('x','iterative',.7)
    with pytest.raises(ValueError):_paired([a,a,b],('bounded','correct'),('iterative','correct'),100,1)
