

#from src.main import monitorDataLambda
import pytest

data = {'bucket':'test-physio-bucket', 'name':'monitorData 03-02-2020.txt'}


###TESTS
#does it run smoothly?

#assert at the end

#do the tables coincide


# do the tables coincide except for ML


class TestNames(object):
  """This class bundles the tests for names. Always leave test_ before test names."""

  def test_bucketname(self):
    name = 'test-physio-bucket'
    print(data['bucket'])
    assert data['bucket'] == name



print('pytest completed')

