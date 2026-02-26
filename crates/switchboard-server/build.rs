fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true) // needed for the in-process mock test server
        .build_client(true)
        .compile_protos(
            &["../../proto/guardrails/v1/evaluator.proto"],
            &["../../proto"],
        )?;
    Ok(())
}
