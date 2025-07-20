.PHONY: clean
.PHONY: test
.PHONY: format
.PHONY: lint
.PHONY: check

all: build

build:
	meson build . \
	  -Dtests=false && \
	  meson compile -C build -v

debug:
	meson build . --buildtype debug \
	  -Db_pgo=generate \
	  -Db_sanitize=address,undefined \
	  -Db_coverage=true && \
	  meson compile -C build -v

setdebug:
	meson configure builddir --buildtype debug \
	  -Db_pgo=generate \
	  -Db_sanitize=address,undefined \
	  -Db_coverage=true

clean:
	rm -rf build

docs:
	doxygen

test: build
	meson test -C build -v

format:
	clang-format -i include/*hpp
	clang-format -i src/*cpp
	clang-format -i tests/*cpp

lint:
	cpplint --filter=-legal/copyright,-readability/casting,-whitespace/braces,-whitespace/indent,-build/include_subdir,-whitespace/line_length src/*.cpp include/*.hpp tests/*cpp

check:
	cppcheck src/*.cpp include/*.hpp
