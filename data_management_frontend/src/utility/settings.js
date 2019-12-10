export const options = {
        sorting: true,
        grouping: true,
        selection: true,
        filtering: true,
        exportButton: true,
    };

export const localization = {
    pagination: {
        labelDisplayedRows: '{from}-{to} sur {count}',
        labelRowsSelect: 'lignes',
        labelRowsPerPage: 'Lignes par page',
        firstAriaLabel: 'Première Page',
        firstTooltip: 'Première Page',
        previousAriaLabel: 'Page précédente',
        previousTooltip: 'Page précédente',
        nextAriaLabel: 'Page suivante',
        nextTooltip: 'Page suivante',
        lastAriaLabel: 'Dernière Page',
        lastTooltip: 'Dernière Page',
    },
    toolbar: {
        nRowsSelected: '{0} ligne(s) sélectionnée(s)',
        addRemoveColumns: 'Ajouter ou Supprimer des colonnes',
        showColumnsTitle: 'Afficher les colonnes',
        showColumnsAriaLabel: 'Afficher les colonnes',
        exportTitle: 'Exporter',
        exportAriaLabel: 'Exporter',
        exportName: 'Exporter sous forme CSV',
        searchTooltip: 'Rechercher',
        searchPlaceholder: 'Rechercher',
    },
    header: {
        actions: 'Actions',
    },
    body: {
        emptyDataSourceMessage: 'Aucun résultat à afficher',
        filterRow: {
            filterTooltip: 'Filtrer',
        },
        addTooltip: 'Ajouter',
        deleteTooltip: 'Supprimer',
        editTooltip: 'Modifier',
        editRow: {
            deleteText: 'Êtes vous sûr de vouloir supprimer cette ligne ?',
            cancelTooltip: 'Annuler',
            saveTooltip: 'Enregistrer'
        },
        grouping: {
            placeholder: 'Déplacez les entêtes pour faire un group by'
        }
    }
};

export const addRow = (newData) => {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            let data_copy = JSON.parse(JSON.stringify(data));
            data_copy.rows.push(newData);
            updateData(data_copy);
            resolve();
        }, 1000);
    });
};

export const updateRow = (newData, oldData) => {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            let data_copy = JSON.parse(JSON.stringify(data));
            let index = data_copy.rows.findIndex(row => JSON.stringify(row) === JSON.stringify(oldData));
            data_copy.rows[index] = newData;
            updateData(data_copy);
            resolve();
        }, 1000);
    });
};

export const deleteRow = (oldData) => {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            let data_copy = JSON.parse(JSON.stringify(data));
            let index = data_copy.rows.findIndex(row => JSON.stringify(row) === JSON.stringify(oldData));
            data_copy.rows.splice(index, 1);
            updateData(data_copy);
            resolve();
        }, 1000);
    });
};