project(
  'quant',
  'cpp',
  version : '0.0.1',
  default_options : [
    'buildtype=release',
    'warning_level=3',
    'cpp_std=c++23',
    'b_ndebug=if-release'],
  meson_version : '>= 0.52.0')

add_global_arguments([
  '-DPACKAGE_NAME="' + meson.project_name() + '"',
  '-DPACKAGE_VERSION="' + meson.project_version() + '"'],
  language : 'cpp'
)

cpp = meson.get_compiler('cpp')
warning_flags = []
foreach cflag: [
  '-Wdouble-promotion',
  '-Wduplicated-cond',
  '-Wduplicated-branches',
  '-Wlogical-op',
  '-Wnull-dereference',
  '-Wrestrict',
  '-Wuseless-cast',
  '-Wshadow',
  '-Wformat=1']
    if cpp.has_argument(cflag)
      warning_flags += cflag
    endif
endforeach

inc = include_directories('include')

subdir('include')
subdir('src')

omp = dependency('openmp')
tbb = dependency('tbb')
eigen = dependency('Eigen3', modules: ['Eigen3::Eigen'])

if get_option('executable')
  quant_cpp_sources = files([
    'src/main.cpp'
  ])

  quant_exe = executable(
    'quantexe',
    quant_cpp_sources,
    cpp_args : warning_flags,
    include_directories : inc,
    link_with : [cppdev_lib],
    dependencies : [omp, tbb, eigen]
  )
endif
