# Aviation Safety Report Knowledge Base

## Environment
Assume you have conda installed,
```
conda create -n ntsb python=3.10
conda activate ntsb
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia pandas
conda install -c conda-forge transformers
pip install wikipedia newspaper3k GoogleNews pyvis requests
```

## Data Accessibility

All datasets are stored under the `data/` directory:

```
data/
├── NTSB_ASRS/       # US NTSB Aviation Safety Reporting System
│   └── avall.mdb    # MS Access database with narratives, events, aircraft, etc.
├── TSB_CANADA/       # Transport Safety Board of Canada
│   └── *.csv         # Public occurrence, aircraft, injuries, events data
├── BEA/              # French Bureau d'Enquetes et d'Analyses
│   └── bea_notified_events.csv   # All notified events scraped from bea.aero
└── FAA_AIDS/         # FAA Accident/Incident Data System (ASIAS)
    ├── a*.txt        # Accident/incident records by time period (TAB-delimited)
    ├── e*.txt        # Edited remarks/narratives by time period (TAB-delimited)
    ├── AcrftSer.txt  # Aircraft make/model/series lookup table
    ├── Airport.txt   # Airport and location lookup table
    ├── aidcodes.doc  # Code definitions and data dictionary
    └── Afilelayout.txt / Efilelayout.txt  # File layout descriptions
```

### NTSB data (US)
Source: https://app.ntsb.gov/avdata
```
wget https://app.ntsb.gov/avdata/Access/avall.zip
unzip DownloadFile\?fileID\=C\:\\avdata\\avall.zip -d data/NTSB_ASRS/
rm -rf DownloadFile\?fileID\=C\:\\avdata\\avall.zip
```

The `avall.mdb` Microsoft Access database format requires `mdbtools` on Linux:
```
sudo apt install mdbtools
```

### TSB data （Canada）
Source: https://www.tsb.gc.ca/eng/stats/aviation/data-5.html

CSV files covering occurrences, aircraft, injuries, events/phases, and survivability.

### BEA data (France)
Source: https://bea.aero/en/investigation-reports/notified-events/

~6500 notified aviation safety events scraped from the BEA search engine. To re-scrape:
```
conda activate ntsb
python scrape_bea.py
```

Fields: file_number, title, summary, date, location, state_of_occurrence, occurrence_class, human_consequences, aircraft_category, manufacturer_model, registration, state_of_registry, operator, operation_type, flight_phase, departure, destination, responsible_entity, detail_url.

### FAA AIDS data (US)
Source: https://www.asias.faa.gov/apex/f?p=100:189:::NO:::

FAA Accident/Incident Data System from the Aviation Safety Information Analysis and Sharing (ASIAS) system. Contains accident and incident records from pre-1975 to present, provided as TAB-delimited text files in zip archives. To re-download:
```
conda activate ntsb
python download_faa_aids.py
```

Data files:
- **A files** (`Apre1975.txt`, `a1975_79.txt`, ..., `a2020_26.txt`): Accident/incident records by time period. Layout described in `Afilelayout.txt`.
- **E files** (`e1975_79.txt`, ..., `e2020_26.txt`): Edited remarks/narratives (redacted per Privacy Act). Layout described in `Efilelayout.txt`.
- **AcrftSer.txt**: Aircraft make, model, and series lookup. Layout in `aircraftseries.doc`.
- **Airport.txt**: Airport and location lookup. Layout in `airport.doc`.
- **aidcodes.doc**: Associated table codes and definitions.
- **10-14-2010-Attention.doc**: Important changes to AIDS data format.


## Build the NTSB Safety Knowledge Graph
