# rt1 validation test: compares the CUDA ray_trace_1 against a numpy oracle.
# The synthetic input + reference .npy files live in tests/data/<halo_name>/ and are
# produced by gen_rt1_reference.py (see the gen_rt1_data target below).

# Built as a CUDA target so nvcc performs the device link against the (separable) core lib.
add_executable(test_rt1 test_rt1.cu)
target_link_libraries(test_rt1 PRIVATE raytracer_core)
set_target_properties(test_rt1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_test(NAME rt1_validation
         COMMAND test_rt1 ${CMAKE_CURRENT_SOURCE_DIR}/data synth)

# Convenience target to (re)generate the synthetic reference data. Requires python + numpy.
add_custom_target(gen_rt1_data
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/gen_rt1_reference.py ${CMAKE_CURRENT_SOURCE_DIR}/data synth
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    USES_TERMINAL
    COMMENT "Generating synthetic rt1 reference data into tests/data/synth"
)
