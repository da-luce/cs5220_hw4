#!/bin/bash

echo "Cleaning up old outputs and logs..."
rm -rf out sim*.log wio_* simfab_traces sim_stats.json simconfig.json || true