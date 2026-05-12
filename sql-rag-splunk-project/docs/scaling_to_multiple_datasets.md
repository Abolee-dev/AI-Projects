# Scaling to Multiple Datasets

To add a new Splunk dataset:

1. Create SQL table DDL.
2. Add dataset schema to `config/dataset_registry.yaml`.
3. Add ingestion mapping.
4. Add business rules.
5. Add few-shot SQL examples for the LLM.
6. Add tests.

Example future datasets:

- incident_events
- change_events
- server_inventory
- application_master
- service_ownership
- patching_status
- observability_metrics
