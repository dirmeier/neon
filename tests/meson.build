gtest_dep = dependency(
  'gtest',
  main : true,
  required : true
)

cppdev_test_cpp_sources = files([
  'test.cpp'
])

cppdev_test = executable(
  'quant_test',
  cppdev_test_cpp_sources,
  cpp_args : warning_flags,
  dependencies : [gtest_dep],
  install : false,
  include_directories : inc,
  link_with : [cppdev_lib],
)

test('quant test', cppdev_test)
