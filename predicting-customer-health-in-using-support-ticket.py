{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3512c528",
   "metadata": {
    "_cell_guid": "29d26fb8-a6c8-4b01-bd18-10dbb4706e43",
    "_uuid": "4b90bdb0-ce72-4ecc-927c-df4f3c374682",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:39.944490Z",
     "iopub.status.busy": "2023-07-04T15:07:39.944059Z",
     "iopub.status.idle": "2023-07-04T15:07:41.752190Z",
     "shell.execute_reply": "2023-07-04T15:07:41.750925Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 1.82831,
     "end_time": "2023-07-04T15:07:41.755549",
     "exception": false,
     "start_time": "2023-07-04T15:07:39.927239",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/kaggle/input/support-ticket-data/support_tickets_dataset.csv\n"
     ]
    }
   ],
   "source": [
    "# This Python 3 environment comes with many helpful analytics libraries installed\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import missingno as msno\n",
    "import pandas as pd\n",
    "\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b6a61828",
   "metadata": {
    "_cell_guid": "7b100b50-81e6-4e65-9b1b-67929fa357e2",
    "_uuid": "cc5eba67-1ac8-4436-b5ed-dd063dc0a9fc",
    "papermill": {
     "duration": 0.01419,
     "end_time": "2023-07-04T15:07:41.784819",
     "exception": false,
     "start_time": "2023-07-04T15:07:41.770629",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Import the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "7f246cd5",
   "metadata": {
    "_cell_guid": "a3085a16-3394-4ab2-b25b-15bbae67ad06",
    "_uuid": "f3fe4d4f-0e8b-4555-9cc4-898e72a39b53",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:41.816067Z",
     "iopub.status.busy": "2023-07-04T15:07:41.815555Z",
     "iopub.status.idle": "2023-07-04T15:07:44.054856Z",
     "shell.execute_reply": "2023-07-04T15:07:44.053557Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 2.258567,
     "end_time": "2023-07-04T15:07:44.057895",
     "exception": false,
     "start_time": "2023-07-04T15:07:41.799328",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['number', 'state', 'active', 'reassignment_count', 'reopen_count',\n",
       "       'interactions_count', 'made_sla', 'requester_id', 'opened_by',\n",
       "       'opened_at', 'sys_created_by', 'sys_created_at', 'sys_updated_by',\n",
       "       'sys_updated_at', 'contact_type', 'location', 'category', 'subcategory',\n",
       "       'u_symptom', 'impact', 'urgency', 'priority', 'assignment_group',\n",
       "       'assigned_to', 'knowledge', 'u_priority_confirmation', 'notify',\n",
       "       'problem_id', 'rfc', 'vendor', 'caused_by', 'closed_code',\n",
       "       'resolved_by', 'resolved_at', 'closed_at', 'days_to_resolve',\n",
       "       'opened_at_new', 'resolved_at_new'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset = pd.read_csv(\"../input/support-ticket-data/support_tickets_dataset.csv\", low_memory=False)\n",
    "dataset.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "87acec5e",
   "metadata": {
    "_cell_guid": "0f13ba98-00c2-4344-b40a-af38cbdc3cd5",
    "_uuid": "67982182-03e0-4ca8-8a04-ead9a387c92d",
    "papermill": {
     "duration": 0.015763,
     "end_time": "2023-07-04T15:07:44.088628",
     "exception": false,
     "start_time": "2023-07-04T15:07:44.072865",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Data Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9e587ab7",
   "metadata": {
    "_cell_guid": "3a6079cf-c02f-4c4c-b656-3f99217a0f39",
    "_uuid": "e0c66c0c-297e-4fb2-97fa-9bed8ad30624",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:44.122593Z",
     "iopub.status.busy": "2023-07-04T15:07:44.121678Z",
     "iopub.status.idle": "2023-07-04T15:07:44.169117Z",
     "shell.execute_reply": "2023-07-04T15:07:44.167810Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.066744,
     "end_time": "2023-07-04T15:07:44.172156",
     "exception": false,
     "start_time": "2023-07-04T15:07:44.105412",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(141712, 9)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Select only the relevant features\n",
    "selected_columns = ['state', 'active', 'reopen_count', 'interactions_count', 'made_sla', 'requester_id', 'days_to_resolve', 'priority', 'impact']\n",
    "dataset = dataset[selected_columns]\n",
    "dataset.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "cba167dd",
   "metadata": {
    "_cell_guid": "c50f3b03-0711-40c3-b49c-37755a9248be",
    "_uuid": "f81f38d8-4b9b-492b-a200-dda125df0522",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:44.211188Z",
     "iopub.status.busy": "2023-07-04T15:07:44.210285Z",
     "iopub.status.idle": "2023-07-04T15:07:44.369891Z",
     "shell.execute_reply": "2023-07-04T15:07:44.368641Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.179865,
     "end_time": "2023-07-04T15:07:44.372997",
     "exception": false,
     "start_time": "2023-07-04T15:07:44.193132",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 141712 entries, 0 to 141711\n",
      "Data columns (total 9 columns):\n",
      " #   Column              Non-Null Count   Dtype  \n",
      "---  ------              --------------   -----  \n",
      " 0   state               141712 non-null  object \n",
      " 1   active              141712 non-null  bool   \n",
      " 2   reopen_count        141712 non-null  int64  \n",
      " 3   interactions_count  141712 non-null  int64  \n",
      " 4   made_sla            141712 non-null  bool   \n",
      " 5   requester_id        141683 non-null  float64\n",
      " 6   days_to_resolve     138571 non-null  float64\n",
      " 7   priority            141712 non-null  object \n",
      " 8   impact              141712 non-null  object \n",
      "dtypes: bool(2), float64(2), int64(2), object(3)\n",
      "memory usage: 7.8+ MB\n"
     ]
    }
   ],
   "source": [
    "dataset.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "6f5392ff",
   "metadata": {
    "_cell_guid": "ecbf8a76-ab8f-456d-b0aa-78441ab2c053",
    "_uuid": "424166bc-86f1-4635-bab2-077263ef0f20",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:44.404806Z",
     "iopub.status.busy": "2023-07-04T15:07:44.404315Z",
     "iopub.status.idle": "2023-07-04T15:07:46.044244Z",
     "shell.execute_reply": "2023-07-04T15:07:46.043230Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 1.65977,
     "end_time": "2023-07-04T15:07:46.047630",
     "exception": false,
     "start_time": "2023-07-04T15:07:44.387860",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAACCUAAAPSCAYAAABGIvJ6AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAADxXUlEQVR4nOzdd1QUZ9sG8Gt26SCCgIoUK/Yae2xRY4s1Yi8xicbeK/ZubFGxd0UFRaPYexdb7BqNvXdUQOmwe39/+O1kVzBvigKr1++c97w4Mzt55pyZ3WeeueZ+FBEREBEREREREREREREREREREX1gmrRuABEREREREREREREREREREX2aGEogIiIiIiIiIiIiIiIiIiKij4KhBCIiIiIiIiIiIiIiIiIiIvooGEogIiIiIiIiIiIiIiIiIiKij4KhBCIiIiIiIiIiIiIiIiIiIvooGEogIiIiIiIiIiIiIiIiIiKij4KhBCIiIiIiIiIiIiIiIiIiIvooGEogIiIiIiIiIiIiIiIiIiKij4KhBCIiIiIiIiIiIiIiIiIiIvooGEogIiIiIiIiIiIiIiIiIiKij4KhBCIiIiIiIiIiIiIiIiIiIvooGEogIiIiIiIiIiIiIiIiIiKij4KhBCIiIiIiIiIiIiIiIiIiIvooGEogIiIiIiIiIiIiIiIiIiKij4KhBCIiIiIiIjJrer0+rZtARERERERERETvwVACERERERERmQURAQAkJiaqyxISEqDRvL21vXPnTpq0i4iIiIiIiIiI3o+hBCIiIiIiIjILiqIgKioKU6dOxfz58wEAVlZWAIA5c+Ygd+7c2Lp1a1o2kYiIiIiIiIiI3mGR1g0gIiIiIiIi+ruePXuGefPm4eHDh3j58iWGDh2KpUuXokePHsiQIQOnciAiIiIiIiIiSmcUMdS/JCIiIiIiIjIDq1evRuvWrQEADRs2xKZNm+Dh4YFZs2ahUaNGads4IiIiIiIiIiIywVACERERERERmQXD7auiKDh48CCqVasGCwsL2NnZYePGjfjqq68gIhARaDScrZCIiIiIiIiIKD3gKA0RERERERGZBUVR1L9jY2MBAElJSXj9+jXOnj2rbsPsPRERERERERFR+sFQAhEREREREZkNRVEQGxuLU6dOoXz58ujTpw8AoH///hg7diwAQKvVQqfTpWUziYiIiIiIiIjo/3H6BiIiIiIiIjI7r169wsuXL+Hj44MNGzagSZMmAICxY8di6NChAACdTgetVgu9Xs/pHIiIiIiIiIiI0ghHZYiIiIiIiCjdel+OPlOmTMiVKxcAoHHjxli7di0AYPjw4Rg/fjyAtxUT4uPj1UDC77//ngotJiIiIiIiIiIiYwwlEBERERERUbokIlAUBQBw7NgxBAcHY/LkyTh+/DgiIiKg1WqRkJAAAGjSpEmKwQRra2sAb6d3KFq0KLZs2ZIGR0JERERERERE9Pni9A1ERERERESUrq1cuRKdO3dGUlISEhMTkTlzZpQuXRrz58+Hh4cHEhISYGVlBQD49ddf0axZMwCAn58fWrZsifnz52PevHlwcHDA77//Dm9v77Q8HCIiIiIiIiKizwpDCURERERERJRuhYSEwNfXFwDQrl07vHz5EtevX8f169eRK1cu7N+/H97e3ibBhA0bNqBp06YQEWi1Wuh0OuTOnRv79u2Dt7c3dDodtFptWh4WEREREREREdFng6EEIiIiIiIiSjf0ej00Gg30ej0AoHHjxjh48CAWLVqEpk2bIi4uDo8fP8ZPP/2EAwcOwMvLC0eOHEkWTNi/fz8WLlyIyMhIeHt7Y9SoUXB3d2cggYiIiIiIiIgolTGUQEREREREROnOmTNnULx4cRQtWhR16tTB1KlTAfwZWhAR1KlTB7t3735vMCE6Ohp2dnZISEiAtbU1AwlERERERERERGlAk9YNICIiIiIiIjIWEBCA0qVLo1WrVgCASpUqAQB0Oh00Gg2SkpKgKAp27tyJmjVr4sGDB6hUqRLu378PKysrJCYmAgDs7OygKAqsra0BgIEEIiIiIiIiIqI0wFACERERERERpSuOjo5QFAXr1q3DtWvXcOfOHQCAodCfhYUFkpKSACBZMOHBgwewtLSETqeDoihpdgxERERERERERPQWQwlERERERESUZgxBA+P///bbb7Fp0ybY2dlBr9dj9+7dAN6GEXQ6nfp3SsGE/Pnz49GjR6yKQERERERERESUTjCUQERERERERGlCRNRqBi9evDBZXq9ePaxZswb29vbYuXMnunXrBuDtFAzvCyaUKVMGsbGxrJBARERERERERJSOMJRAREREREREacIQHpg5cyby5cuH48ePq8veDSbMmzcP/fr1A/D+YMKJEyfw7NkzZMuWTV1PRERERERERERpi6EEIiIiIiIiSjNJSUk4dOgQIiIi0LZtW5w8edIkmFC3bl2sXr0adnZ2mD59+nuDCYa/3dzcoNfrOX0DEREREREREVE6wVACERERERERpRkLCwssX74c3333HW7fvo1mzZolCyYYKib8VTDBOISg0fBWl4iIiMyHXq9P6yYQERERfVSKiEhaN4KIiIiIiIg+b1FRUejSpQsCAwPh5eWFtWvXomzZsjDcsiqKgq1bt6JFixaIiYlB//79MXny5DRuNREREdE/JyJqADMler2eIUsiIiL6pDCUQERERERERB/du4PrOp0u2RQLfzeY0LZtW0RGRmLUqFEYMWJEqh4HERER0X9h3Cc6e/YsLl++jEOHDqFw4cLw8vKCr69vGreQiIiI6MNjKIGIiIiIiIhSzfHjx1G8eHHY2tqm+Bbgu8GE4OBglCtXziSYsH79egwePBi7d+9Gjhw50uAoiIiIiP454woJq1atQv/+/fH8+XOTbVq2bImBAweicOHCyQKcREREROaKoQQiIiIiIiJKFWvWrEGrVq3QoUMH+Pv7w9bWNsWKCREREejUqRPWrVsHLy8vrF69Gl9++SVEBCICjUaDuLg42NjYICkpCRYWFml0RERERET/3OrVq9G6dWtYWVlh6NCh8PHxQVxcHPz9/XHhwgVUrlwZ/fr1Q926dTmNAxEREX0SOHJDREREREREH52IwMrKCq6urli8eDEsLS0xderUFIMJTk5OGDRoEC5duoSrV6+iZcuWasUEAxsbGwBgIIGIiIjMytmzZ9G/f39oNBosX74cLVq0UNdZWlqia9euOHz4MNq0acNAAhEREX0y2KshIiIiIiKij05RFNSvXx9Lly6Fp6cn5s2bh/79+yM2NhZarRY6nc5k+xIlSsDNzQ0A8ODBA1SrVg1nz55VSx4TERERmaPz58/jyZMnGD16tEkg4dixY5g2bRrevHkDPz8/dOjQAQDUKaxY8JiIiIjMGUMJRERERERE9NEYD6BbWlqiVq1amD179l8GE+Lj46EoCooXL44ff/wRDRs2REJCArJmzZpWh0FERET0nxj6RFu3bgUAlC9fXl134sQJdO3aFefOnYOfnx8mTJigrgsLCwPwNuDJYAIRERGZK4YSiIiIiIiI6IN5d7D83coGKQUTBgwYoAYT4uLiYG1tDeDtoL1Op0NQUBCe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HN999x08PDwwa9YsNGrUKK2bSZTqdu/ejXr16sHR0RFz585Fs2bN1HXGlRQiIiIQEhKC+fPn49SpU9BoNBAR2NraIiYmBvnz58euXbvg5eUFnU7HNwWJiIgo3TLu4xw/fhxjx459bzABAF6+fAl/f38cPHgQoaGhAIC6deuicePGsLe3x48//oivv/4aGzduTO1DISIiIkozDCUQEREREaXg2bNnmD17NsaPHw9HR0e8fv0anp6emDt3LurVqwfAdICS6FOm1+sRExODn376CcHBwVi4cCE6dOigrktp3uOEhAS8ePECkyZNwu+//47Lly+jRIkSKFGiBHr16oUsWbIwkEBERERm4a+CCaNHj0apUqUAAImJibC0tERSUhJiYmKwaNEiBAUF4dKlS0hKSkLOnDnx8uVLvH79GkuXLsX333+fhkdFRERElHoYSiAiIiIi+gt169bFzp07odFo0K1bN8yYMQPAnwOORJ+LsLAwlCpVClZWVrh8+TKsrKzeG0gATAfvk5KS8PLlS5MgAgMJREREZE7+bjDh3T7OzZs3cfbsWQwbNgwvXrxAREQEvLy8cPToUXh6eqbJsRARERGlNou0bgARERERUXq1e/du7NixA/b29oiOjsbatWtRqlQptGnTBpaWln/5QJboU/Po0SM8e/YM+fPnh4WFBUTkL89/vV6vhg8sLCzg5uYGAOpnGEggIiIic6IoihpMKF++PIYPHw4A2LFjBwCowQStVgvDe4CKoiBPnjzIkycPKlSogIMHD2Lz5s2YNGkSPD09GdIkIiKizwZHUImIiIiI3sPHxwft2rXD/PnzMW7cODx9+hT9+/fHypUrAbx9uKrX69O4lUSpw8LCAlqtFrdv38adO3feO3WJXq9HfHw8Zs+ejQcPHqgD7YYwAqc8ISJKn+7evZvWTSBKl4wLDRuCCQDUYELt2rWxY8cOjBw5EqdPn1a3e3cfHh4eaN26NVavXo0cOXIgKSmJgQQiIiL6bDCUQERERERkREQgItDr9ciZMyfmzZuH1q1bo0uXLhgyZAieP3+OgQMHJgsm6HS6NG450cdVuHBhfPXVV4iKisKyZcvw5s2bZNvodDpoNBrExMTg559/xujRo5GUlJQGrSUion9i0qRJKFu2rPrGN9HnzjiIEB8fj/DwcNy+fRsJCQkm25UvXx7Dhg17bzDBuGKC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7j1crvHuH/+9dg52BDnIAUEiJA1xLCItQtQhraqrpKVF61BP0cO3aKnDo6mmtFqHpyqlDg+igmjrEJpiJxJxCEGqjkkQQohoVMg52ffvD7893TsJ7dPOvdda5vW6rrkys2ZmfT4r18yetda8575XrFgRv/a1r8Usy+I555xTua/5Z0jTSftzzjkndujQIW666aaxc+fOggmUSvPX9O9///u4yy67xDZt2sQsy2LHjh3jt771rThlypTKY1b/pWyMH4zGec4558Q99tij8qVs0z6VYwDK6KqrropZlsUf/ehHlSl9rr766hbBhDlz5rR4zpgxY+LRRx8de/ToETt27Bh32mmneOKJJ1aCm44PKIPmf7NvvvnmuPvuu8fu3bvHIUOGxNNPPz2++eabMcaW506bj5jw+c9/Pn7zm9+MV1xxReUzYcstt6wEeXwmAP8OoQQosMbGxrhgwYIY4wcHi00nSiZOnBj/9re/xRhj/Pvf/x5feumlyg7xcccdF+vq6uLo0aMrz4sxxldffTV27969ckC6+eabx3fffbe1Nwmq6t57741t2rSJnTp1+tA5/dYWTOjSpUscNWpU5UvbDTfcME6fPr212oaqap58b3597NixlYPJMWPGrPGYps+Nyy+/PGZZFr/97W+vse7mJ1vOPvvs+MQTT1S9f/hPNb2mb7zxxphlWVx33XVjXV1dPOWUU+KsWbM+9HkzZ86M11xzTRw8eHDcbLPN4nrrrRf322+/eO6551aGZHXCkY+T8847L2ZZFi+++OIY4z9CbDHG+P/+3/+LWZbFAQMGxCzL4o9//GMnGimlD9svuvXWW/+l/aKf/exnsXv37vHnP/953GSTTeKGG24omEApNP+bfc0111Re70cccUT8/ve/H4cNGxbbtGkTd9ttt3jnnXeu9XlN74cVK1bExYsXVwL8vXr1ii+//HKrbQtUy8MPPxy7d+8e11133XjNNddUlq8tmLD6iAmrVq2Kb775Zpw1a1Z8//33K+eWHB9QNr/73e8qnwkbbLBBXGeddWKWZXGnnXaqBPGbnzttPmJClmXxU5/6VNx5553jN77xjcoos94HwL9LKAEK7MYbb4xf/epXWyTZx4wZE7Msi4cccsgaw+09++yzsU2bNnHzzTePc+bMaXFAGWOMQ4cOjccff3zcYostYpZllfnDoYzmzp0b99xzz5hlWfztb39bWb62IfSa71wfffTRLXaut9xyy8qUDlAWq494sGLFirhy5co1Xv9NUy9sscUW8Q9/+MNa13XQQQfFLMvijTfeuMa6Y3SwSXncc889cbvttosnn3xy3HHHHWNdXV086aST4owZMz7yecuWLYvz5s2LM2fOjDH+4zVvSFbK4F8JDqxatSq+//77cdddd41ZlrWY1qTpeOLSSy+NAwcOjBdccEHceeed4+zZs5P1DNVWzf2iffbZJ37605+OixYtiv/zP/8Te/XqVQkmvP3220m3A6rhzjvvjPX19bFnz57x+uuvryw/44wzKlOy9e/fP44bN65y3+qfJc1vH3HEETHLsnjdddfFGO0fUWyrvz6bfvU9duzYNR7zz4IJXut8HLzyyiuxX79+caONNoqjR4+OL730Urzzzjsr0xluvvnm8ZVXXokxtjx3eskll8Qsy2J9fX0cMWJEjPEfnw3OEQH/CaEEKKj3338/nnrqqTHLsrjLLrvEl19+Od51110xy7LYvXv3eMMNN6zxnJdffjl27do17rrrrpVlixcvrlxvGn5y2bJlUu6U3rPPPhs7d+4c99tvv8qyjzpobL5zffXVV8cRI0bEn/70p5WdbyiL5icJ77vvvvjtb387Dho0KA4aNCgee+yxLX759Nxzz1VGCFlnnXXipZdeGl977bW4ZMmSuGjRovjd7343ZlkWBw8eXBmBB8qo6X3xzjvvxBg/mNpnu+22i3V1dfFb3/rWGsGEtQUPmpb5dThl0fy1+pe//CU+8sgjcdy4cXHRokVrfY0feeSRMcuyePvtt8cYPzgZ3+TTn/503GmnnWKM/wgqrD6PLBTRv7tfVF9fH3/961/HefPmxeXLl8elS5dW5ko+/vjjY4wffKY0BRM22mijeNZZZ9lfIneXXXZZHD9+/Frve+mll+LgwYNj27Zt47XXXltZ3jRSzvrrrx+HDx8esyyL/fv3bxFS+7BgctNIVF/4whcSbA2kce2118Y//vGP8eSTT46DBg2qLG96nTf9+6+MmABl9tBDD8Usy1qMFBJjjO+991484IADPjKYcNFFF1V+0HXFFVdUlgvsAP8JoQQosOeeey4ecsghlV9zZ1kWe/ToEX/3u99VHtP8wPHVV1+NPXr0aDEsa5PTTz89ZlkWr7rqqtZqH5K64YYbYpZl8eijj44xOnFObWj+N/9///d/Y9u2bStTknTu3LlywHjuuedWvpydPn16/K//+q/Kff3794+77LJL7N+/f+XzpWnIPgeXlMFHhQaajyJ16623xu23336NYELz599zzz2V1z+U1ejRo2OPHj3iBhtsELMsi5/5zGfir371q/j+++/HGP+xj9T0pdSmm24an3322bhkyZIY4z+mbvjmN78Zly9f7rOA0qjGftHAgQPjpz/96bjttttW9ouahiaOMcaFCxfGSy+9NHbo0CH27dvXNA7k6p577qkMvz1hwoQ17r/99ttjlmWVX7XGGOMvfvGL2KZNm7j++uvHJ598Mv7973+vjJS2zTbbxNtuu63y2ObvqaZQwssvvxzXX3/9OGTIkIRbBtUzderUyt/47bffPn7uc59b6+M+LJhw9tln+/EKpbS24+R77703Dhw4sHJcsHLlyhYjHfyzYELTiAlZlrWYNtfxAvDvEkqAAmtsbIwLFy6MgwYNim3atInt2rWL5557buX+tX0JO3r06JhlWayrq4snn3xyvOaaayq/itp6660lfvnYaAol7Lvvvv80kLBo0aJ4//33x/fee6/Fcr+Gpaz+8Ic/VE5IXnHFFXH+/PnxzTffjDfccEPs2LFjzLIsHnPMMZXRct577704atSoOGDAgNitW7eYZVn8xCc+EY866ihzAlIqzf9uT5o0KV500UVxxIgR8be//e1a/6avHkx4/vnnK/eddtppsUuXLvH88893UoXSuvXWWysnCg844IDYp0+fuO6668aOHTvGb33rW3HhwoUtHt/0RVT37t3joEGD4g477FAJrM2dOzenrYD/zL+7X9S/f/+44YYbxizL4sYbbxz333//OGfOnBjjB/tFTZ8r7777brz88svjCy+8kNs2QpNjjz02ZlkWu3XrFhsaGlrcd8stt8Rjjz22MtXILbfcEnv27Bk7duwYH3nkkRjjB+eRxowZE9u3bx/bt28ft9lmmxZTOayu6Qcuw4cPX2MKUSiqb3/725X9o2222eZD/36vHkzYeuutY5Zljg8onebHwtOmTYu33357vOWWW+LPfvaz2LVr1zWmcG5+HvX/Eky4/PLLE28J8HEnlAAFN378+JhlWWzXrl1liO3HHnvsQx+/ePHi+Itf/KKys9B06dWrV3zuuedasXNIa8aMGbFXr16xf//+lXnAVw8nNH3JOn369DhkyJAPnTsWymTOnDmVecGbj5wT4wcj7DSNrHPaaaet8dxXX301zpw5MzY0NMS5c+dWTs4LJFA21113XWzTpk2LfZ1999033n///S2GpI+xZTDhsMMOi2PGjInHH3985de0TV9AQZk0NjbG999/Px544IGxa9euccyYMTHGGF988cU4atSo2Lt375hlWfzGN76xRjDhiCOOiBtttFHlPTB48OAWX8RCmfwn+0WvvPJKfPbZZ+Mdd9wRn3vuucp7pfn7YPWhviEvzY91m0b7WFsw4aWXXqq8Xo855pjYvn37ymdE83VsvfXWlX2pzp07x/vuu2+NmhMmTIjdunWLHTt2XGMqLCii5n+/m6YqXGeddeJvfvObD31O82DCr3/967j77rsbSY3SGj16dFxvvfUqx8ibbrpp7NKlS7zuuuvWCNqsLZjQt2/f+NJLL62x3ubBhKuvvjr5dgAfX0IJUDCrn+yYPHlyHDJkSLzwwgvj4YcfHrMsi7vsskucOnXqR66noaEhHnfccfFrX/taHDFiRHz55ZcTdg3V989O/M2fPz/utddelV9tNGnayW6+c/2lL32pxRzKUGZPPfVU7NixYzziiCNaLJ8yZUrccccdY5Zl8ayzzlrrc9f2ZZOT7JTNnXfeWTkhctJJJ8Uf/vCHcauttopZlsVtt902jh07thK4aXLbbbfFPffcs0WIYcCAAZVfgvgiljJasGBB7NatW/zRj37UYvmiRYvi3XffHfv27fuhwYSnn3463nXXXXH69OmVYe29Dygj+0XUkuav2W984xuVYMLaAgUvvvhirKuri3379o2vvfZa5Ti5Kby55557xq9+9avx2GOPjRtuuGGLaUuavPTSS/Hss8+Ozz77bKItguprPqLHySefHLMsi+uuu+5Hng9q+tu/bNmyFsPcQ5ncfvvtsa6uLmZZFg877LB40EEHVUaM2meffeLs2bPXeE7zc6cHHnhgzLIsDho0KK5atWqNfaLzzz8/rrPOOvHpp59Ovi3Ax5dQAhRI8w/7Z599Nl511VVx9uzZlYPHF198MR5yyCExy7K46667rhFMaNqRaFrPokWLYozmeaJ8mr8XZs2aFe+55574m9/8Jt54441xyZIlcfny5THGGJ988sm4/vrrxyzL4le+8pW4ePHiNV7vZ599dsyyLA4dOjS+9dZbrbod8J9a29/vm266aY0T7A8//HBlCO4zzjijxeNfeuml+OCDDybvFVJZ/X3wzW9+M3bs2DGOHTu2suy1116LRx99dGzfvn0cMGBAvOWWW9YIJjz22GPx5z//eTzwwAPjGWecEd94440YoxOOlMPqJwUbGxvjokWL4mabbVaZV7z5ScUVK1b802BCc44XKAP7RdS6xsbGFvstJ510UsyyLHbt2nWNERNmzZoV27VrF/v27Vs5Dm4KJDQ2NsaNN944fv3rX49vvfVWXLBgQYxx7ftETcfeUCSr7xctWrToQ1+rp556asyyLHbs2DHecccd//I6oehW3y865phj4vrrrx9vueWWyrInn3wy9unTJ2ZZFj/72c+u9bxo82OII444ojIa7dr87W9/q0LnQC0TSoCCaL7zO378+Dhw4MCYZVn8whe+UJnfddWqVXHGjBnx0EMPrQQTms8L2OSGG26oHFSuvm4ouuav17vuuiv279+/xS9bP/nJT8ZRo0bF+fPnxxhj/NOf/lQZmmzo0KHxvPPOi/fff38cN25cZXSRTTbZxHCTlE7z98L06dMrt8eOHRuzLItHH310jDHGhx56aK0n3ptOOl522WWxT58+8amnnmrF7uHf8/jjj3/ofVOmTIkzZsyI/fr1i8cff3xledOvod5444347W9/+yODCTF+sM/UdAJHIIEyaP558Kc//Sl+//vfjwcccED8+c9/Hvv27RuvuOKKGOOa01itHkw4/vjj43vvvRdj9NqnfOwXUeuaf/n06KOPxjvuuCNecMEFsVevXpURE5pCak2aRok666yz4ttvv11Z3jSs/W9/+9vKMueNKIvmr9U77rgjHnPMMbFfv35x8ODB8dhjj4133313ZRSoJv9qMAGK6qOOkx988MH4/PPPxy222CKeeOKJleVNQZ3nn38+br/99pVgwrx589ZYx9qOIwBSEEqAghk7dmxs3759zLIs/uAHP4gzZ85cY0dg9WDCww8/XLnvvPPOi1mWxYMPPtgvnii12267rcXw3GPGjIlnn3127NixY+zdu3c8/fTTKydWpkyZUjnh3jRnYNP17bff3nCTlNo111wTsyyLp59+eowxxrlz58Y+ffrET37yk3HMmDGVoYnXduJ9+fLlceDAgXG77bYzUgiF98tf/jJmWRYvvPDCNe675ZZbYpZl8bjjjouDBw+Oo0aNijH+I5DQdHLyo4IJTrZTdtdee22LoGbT5ZBDDqk8ZvWwwerBhMMOO6wyLDGUkf0ialHzfZhrr702duvWrXI+qGvXrnGjjTaKWZbFHj16tAgmjBkzJm666aaxa9eu8Qtf+EL81a9+VRmee4cddvA+oNSa7xe1bdu2ci514403jkceeeQaU5IIJlBW/8px8jHHHBN32223eOWVV8YY/3Gc3PTdwPPPPx+32267jwwmALQGoQQokHvuuSdmWRY7deoUr7rqqg99XGNjY5wxY0ZlKoctt9wy3nTTTZWh+zbccMP417/+tRU7h+p64IEHYvfu3WOnTp0qO9QxxnjhhRdWDjQ32GCD+P3vf79yIuXFF1+Ml19+eRw+fHjcf//945e+9KV4xRVXrHVuTCiLhx9+uPJeuOyyy2KMMb777rvxiCOOqPwiavUhi5csWRJj/OCz4uijj45ZlsWzzz7b0KsU3sMPP1w5sXjPPfe0uO+Pf/xj3GyzzWLbtm0r4YTVfVgw4dZbb61MaQVl9dBDD8WOHTvGTp06xV/84hfxsssui9/85jcr74kf/vCHlceuLZgwfvz4uMEGG8QePXoYdpXSsl9ErfvDH/4QsyyLG220Ubz++utjY2NjfPnll+PEiRPj/vvvX3kfNE3l8M4778T/+Z//iVtvvXWLMNs222wTX3311Rij6XsopwceeCCuu+66sVu3bvHyyy+P06ZNi/fff3886qijKkPV77HHHnHOnDktntcUTOjcuXOLqeCgyP4vx8n/9V//tcbz1xZM+NznPieYBuRCKAEK4q233op77LHHGkPoNR8lYeHChfG1116r3H7llVfikUce2eLgcsstt/SrcEptwYIF8ctf/nLMsiz+6le/qiy/8MILY7t27WL79u3jmWeeGfv06RM7deoUTz311DUSvk6sUFarv3YvueSSmGXZGidMpk+fHrt37x6zLIvbbbfdWl/zTSdc9txzz8p0J1B006ZNi1//+tfXet+dd94ZBw4cGNu0aRN33nnn+MQTT6zxmNWDCR07dozdunWLd955Z8q2oepW/7v+q1/9KmZZ1mKO2HfeeSdec801sU2bNjHLsvijH/2oct/qwYTly5fHCRMmVMKa9pUoA/tF8IHGxsa4ePHiuN9++8Usy+L111+/1sd97Wtfi1mWxe7du8f77rsvxhjjokWL4vPPPx/POOOMeNppp8WLLrqo8kWUqXwoi9X/rl9wwQVr7BfF+MF897fcckvcaaedYpZl8aCDDlrjb/5pp50WsyyL/fr1qwTXoOj+1ePkXXbZJT755JNrPGZtwYRPfepTLab2AWgNWYwxBiB3L7/8cthpp53CDjvsECZNmlRZvnTp0vD666+HM888M8yYMSPMmjUrfOc73wmHHnpo2HnnnUMIIVx44YVhxowZoWvXruGEE04IvXv3zmsz4D/21FNPhaFDh4bhw4eHK664IoQQwm9/+9twyimnhGXLloUHHnggDB48OJx99tnhoosuCl26dAmHH354OPPMM8OGG24YGhsbQ11dXQghhBhjyLIsz82Bf8uoUaNCt27dwiuvvBLuvPPOMHny5BBCCKtWrQp1dXUhy7LwwAMPhGHDhoUlS5aEffbZJwwZMiTsscce4e233w5XX311uPfee0OfPn3C5MmTw6abbtrivQFl8Otf/zosXrw4nHbaaZVld9xxRzj99NPDCy+8EI455phw7rnnhl69erV4XtPf/nnz5oXTTz89TJ48OTz44INh4403bu1NgP9Y0+fBm2++GcaNGxfuueeeNfZvbrrppnDkkUeGxsbGcM4554QRI0aEED74zGjTps0a6/yw5VBU9osghL/97W9hwIABoa6uLjz77LMtjn1XrlwZ2rZtG0II4Qtf+EIYN25c2HDDDcNNN90UPvOZz6x1fT4LKKNRo0aF7t27h1deeSWMHTs2TJ06NYTQ8vW8cuXK0NDQEP7f//t/Yfbs2eHnP/95+O53vxtWrFgR2rdvH0II4Sc/+Uk46qijQp8+ffLaFPi3fdhx8mmnnRZmzJgRjj322PDjH/94jePkps+MGTNmhKFDh4Y333wzvPnmm6F79+6tvQlADRNKgIJ46qmnwo477hj22muvcPfdd4cOHTqEl156Kdxwww3hmmuuCa+++mro3r17mD9/fmjTpk34yle+En75y1+GHj165N06VNUbb7wRfvrTn4aTTjopbLPNNmHSpEnhxBNPDDNnzgzjxo0LBxxwQAghhJdeeikMGzYszJo1K2y00Ubha1/7WvjBD34QNtxww5y3AP4zb775Zthkk01CCCH069cvbLLJJmHixIktHtN0MPnYY4+F73znO+GZZ54JixYtCm3btg0rV64M7dq1C0OHDg1XX3116NWrl5OOlM78+fNDz549QwghXHLJJeG73/1u5b5x48aFU045JcyaNSuccMIJ4eyzz668Z5o0fWk7f/780LZt29ClSxfvA0qn+efBwIEDQ69evcL48ePX+tj/azABysJ+EXzgvffeC9tuu21YtWpV+Mtf/hI23HDDFiG1ptf166+/HoYNGxaeeeaZ0LVr13DLLbeEffbZp/JYwX3KqvnnwYABA8L6669fCSWs7v333w+XXXZZOOOMM8J+++0X/vznP4cQQosAz9puQ9H9p8fJTftML730UujQoUPYeOONBTWBVuWvDRRE3759w+677x4mTZoUTj311HDOOeeEffbZJ4wYMSJsuOGGYeTIkeHZZ58NN954Y9h4443DDTfcEJ577rm824aq23jjjcPPf/7z8IlPfCKEEMKDDz4YXnjhhfCTn/wkHHDAAaGxsTGsWrUq9O3bN3z5y18O7du3D+uss0644IILwq9+9avQ2NiY8xbAf2ajjTYKEyZMCCGEMHPmzLBkyZLwyiuvhPjBtFshhBDq6upCY2Nj2HXXXcPYsWPDmDFjwnHHHReOOuqocPLJJ4c//vGP4eabb3bindLq3r17eOSRR0JdXV343ve+Fy655JLKfZ///OfDxRdfHLbaaqtw+eWXh5EjR4a5c+e2eH7TSffu3buHLl26hBij9wGl0/zz4JlnngkLFy4Mr7/+eovPgyaHH354uOGGG0JdXV0499xzw7nnnhtCCF73lJ79IvhAXV1d6NixY5g7d2648sorQwj/2N8J4YO/9zHG0K1bt7DJJpuELMvCggULwr777hseeuihShBBIIGyav558Pzzz4f58+eHJ554Yo19ohBCWG+99cJhhx0WNthgg3DvvfeGv/71ryGEsEYAQSCBsvlPj5Ob9pn69u0bNt5448qoUwCtpjXnigA+2nPPPRe33XbbmGVZzLIs1tXVxRNPPDG+/PLLcenSpZXHHXXUUTHLsnjdddfl2C2kt3DhwrjrrrvGNm3axAceeCDG+MF8mitWrIgxxnjWWWfFTTbZJI4aNSpuv/328bnnnsuzXaiqiRMnVj4PLr300sryxsbGf3kd5gyn7B577LHK++Diiy9ucd+4ceNiv379YpZl8cQTT4yvv/56Pk1CYs0/D37zm99Ulq/t82DMmDGxvr4+ZlkWL7zwwtZsE5KyXwQxXnXVVbF9+/Zx8ODBccKECZXlTe+D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P/vZz+q1ZoNKCbfccktD3nZA9957byxfvjyOP/74uP766+Ohhx6q0/tGjhyZ8Sxf9umnn8aTTz4ZERGdO3eOM888c5/XnXvuuV971owZMw54vkWLFjFlypTo3r17RETMmTNHKQEAAAAAAADIiPXr18ekSZOitLQ0Vq5cGS1btqw9N2nSpBg9enRMnDgxpk+fnmJKkjJ37tyYO3du9OrVK55//vlo3rx5RETcfvvt0aNHjxgzZkx8//vfj9LS0jqvmRX7/r/33ntx8803R0TElClTEmumNNTs2bNj+/btEbH/XRIaU7du3aJt27YREfXahQEAAAAAAADgQCorK6O6ujp69eq1VyEhIuKCCy6IiIgNGzakEY1G8MQTT0RExA033FBbSIiIaNu2bYwePTp27NgR5eXl9VozK0oJP/nJT2Lr1q0xdOjQ6NOnT9pxvmLmzJkR8cV2G9myK8HOnTsjIqKwMCv+EwIAAAAAAACHgC5dukTTpk3jpZdeis2bN+91bt68eRER0bdv3zSifW3r1q2LadOmxa233hrTpk2LdevWZXTtioqKiIh45ZVXMrp2Y3r//fcjIqJTp05fObf72IsvvlivNRt0+4ZMmjVrVsybNy/atGkT99xzT9pxvmLt2rXx0ksvRUTEmWeeGZ07d045UcSKFSti06ZNERFxwgknpJwGAAAAAAAAOFR885vfjNtuuy3Gjh0bXbt2jf79+8cRRxwRr7/+erzwwgsxcuTIuOaaa9KOWW/z58//yvfRs2bNirFjx0a/fv0ysnZNTU1EfPF97vDhwzOydmNr165dRHzxPXnXrl33Ord27dqIiHj77bfrtWaqpYSPP/44fvGLX0RExJ133hnt27dPM84+zZw5s/Z/nmy4dUPEF/fr2O3iiy9OMQkAAAAAAABwqBkzZkwcddRR8eMf/zimTJlSe/yMM86IK664Ipo0aZJiuvpbt25d3HPPPVFdXf2Vc3fffXds2LAh2rRpU6e1dn93XFBQEBERH330UTz88MO1x3dfU1NTE7/97W+je/fucfTRR2fgt2gc5513Xjz22GNx5513Rt++faO4uDgiIj788MOYNGlSRER88skn9Voz1VLC2LFj4/3334/TTz89rr766jSj7Ncf/vCHiIho3rx5DB48OOU0EXPnzo05c+ZERMQpp5wSAwcOTDkRAAAAAAAAcCiZMGFC/OY3v4nx48fHsGHDok2bNvHPf/4zfvnLX0afPn1i9uzZMWDAgLRj1tn8+fP3e66mpibKy8sTmz1v3rys/S58Xy677LIoLy+PRYsWRffu3aNfv36xa9eueOKJJ6KkpCQiIg477LB6rZlaKeGvf/1rTJ8+PYqKimLq1Km1TZJs8vLLL9fe9+Oiiy6KVq1apZpn9erVceWVV0bEFyWJmTNnNujvtnjx4gwng9y1YMGCtCNA6nwOwOcAInwOIMLnACJ8DiDC5wAifA4g37344otx0003xejRo+OGG26oPd6rV694+umno3PnzjF69OicKiWsX79+v+cKCwujd+/ecdNNNzVo7VtvvTUWL168z10YDjY7GxUVFcX8+fPjzjvvjEcffTQefPDBaNWqVfzwhz+MMWPGxHHHHVd7i4c6r5lQ1gPasWNHjBw5MmpqauLnP/95nHjiiWnEOKiZM2fWPh8+fHiKSSL+97//xXnnnRebN2+OgoKCeOihh+I73/lOqpkAAAAAAACAQ8szzzwTERF9+vT5yrl27dpF9+7d45VXXomNGzdG27ZtGzteg9x0000NLh2kuXZamjVrFuPGjYtx48btdXz3P37v0aNHvdYrzFSw+rjttttizZo10bFjxxg/fnwaEQ5qx44dMXv27IiI6NChQ5xzzjmpZfnoo4/ie9/7XlRWVkZExL333huXXXZZankAAAAAAACAQ9POnTsjIuKDDz7Y5/ndx5s1a9ZomcgOjzzySEREXHrppfV6Xyo7Jdx1110REXH22WfH008/vc9rtm7dWvtz1qxZERHRvn376Nu3b6NkfOqpp+Ljjz+OiIjLL7+83vfFyJTNmzdHv379YtWqVRHxxfYf11xzTSpZAAAAAAAAgENbr1694ne/+11MnDgxBg4cuNct7h9++OGoqKiIU045JY444ogUU5KkTZs2RcuWLfc6NmfOnJg+fXr07Nmz3rfuSKWUsLtdU15eHuXl5Qe8duPGjbW7Apx11lmNVkrIhls3bNu2LS688MJ47bXXIiJi7NixceONN6aSBQAAAAAAADj0DR48OB544IFYvHhxdOnSJfr37x9t2rSJlStXxvPPPx/NmjWLSZMmpR2TBJ122mnRsWPH6Nq1axQXF8eyZcti8eLF0blz5/jjH/9Y73/Qn0opIdtt3LgxFixYEBERZWVl0b1790bPsGvXrhg4cGAsWbIkIiJGjRoVd999d6PnAAAAAAAAAPLHYYcdFgsWLIh77703Hn/88Xjsscdi586dUVJSEkOGDIlf//rX0a1bt7RjkqBLLrkk/vSnP8Wrr74au3btik6dOsWNN94YY8eO/coOCnWRSimhpqbmoNeUlpbGe++9F8ccc0xUVlYmH2oPjz76aOzatSsi0tkloaqqKoYMGRLz58+PiIihQ4fG/fff3+g5AAAAAAAAgPzTrFmzuO666+K6665LOwopGD9+fIwfPz5j6xVmbKVDyO5bNxQVFcWQIUMadXZNTU1cffXVMWfOnIiIGDhwYJSXl0dBQUGj5gAAAAAAAACAr6tBOyUsXbo0Kioqal9v3Lix9nlFRUXMmDFjr+tHjBjRoHB1UVFREUuXLt3r2JYtW2p/fjlLv379okOHDvtd76233orly5fXXtu+ffs6Z9myZUttmWDPfLvNmTMn2rZtW/u6rKwsysrK9rp+zJgxUV5eHhER3bp1ixtuuCHeeuutA861PQoAAAAAAAAA2aigpi73UviSESNGxMMPP1zn6xswos63b5gxY0ZceeWVdV530aJF0bt37/2e/9WvfhV33XVXRETMnj07Bg8eXOe1Kysro1OnTnW+fty4cV/Z9mL3710fDfn7AgAAAAAAAEDS3L5hD9XV1fHII49ERETr1q2jf//+KScCAAAAAAAAgNzVoJ0SAAAAAAAAAAAOxk4JAAAAAAAAAEAilBIAAAAAAAAAgEQoJQAAAAAAAAAAiVBKAAAAAAAAAAASoZQAAAAAAAAAACRCKQEAAAAAAAAASIRSAgAAAAAAAACQCKUEAAAAAAAAACARSgkAAAAAAAAAQCKUEgAAAAAAAACARCglAAAAAAAAAACJUEoAAAAAAAAAABKhlAAAAAAAAAAAJEIpAQAAAAAAAABIhFICAAAAAAAAAJCI/wdaEqmDO+Bi7AAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 2500x1000 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# check for empty values\n",
    "msno.matrix(dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e7143307",
   "metadata": {
    "_cell_guid": "1d202fc2-c05c-432f-9ec2-2d5ef4f44946",
    "_uuid": "45d542c1-bfa4-4690-9642-1649ff16d3d0",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:46.084663Z",
     "iopub.status.busy": "2023-07-04T15:07:46.083202Z",
     "iopub.status.idle": "2023-07-04T15:07:47.704844Z",
     "shell.execute_reply": "2023-07-04T15:07:47.703813Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 1.64439,
     "end_time": "2023-07-04T15:07:47.708112",
     "exception": false,
     "start_time": "2023-07-04T15:07:46.063722",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAACCUAAAPSCAYAAABGIvJ6AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAADT2ElEQVR4nOzdd1QUZ9sG8Gt26SCCgIoUK/Yae2xRY4s1Yi8xicbeK/ZubFGxd0UFRaPYexdb7BqNvXdUQOmwe39/+O1kVzBvigKr1++c97w4Mzt55pyZ3WeeueZ+FBEREBEREREREREREREREREREX1gmrRuABEREREREREREREREREREX2aGEogIiIiIiIiIiIiIiIiIiKij4KhBCIiIiIiIiIiIiIiIiIiIvooGEogIiIiIiIiIiIiIiIiIiKij4KhBCIiIiIiIiIiIiIiIiIiIvooGEogIiIiIiIiIiIiIiIiIiKij4KhBCIiIiIiIiIiIiIiIiIiIvooGEogIiIiIiIiIiIiIiIiIiKij4KhBCIiIiIiIiIiIiIiIiIiIvooGEogIiIiIiIiIiIiIiIiIiKij4KhBCIiIiIiIiIiIiIiIiIiIvooGEogIiIiIiIiIiIiIiIiIiKij4KhBCIiIiIiIiIiIiIiIiIiIvooGEogIiIiIiIiIiIiIiIiIiKij4KhBCIiIiIiIjJrer0+rZtARERERERERETvwVACERERERERmQURAQAkJiaqyxISEqDRvL21vXPnTpq0i4iIiIiIiIiI3o+hBCIiIiIiIjILiqIgKioKU6dOxfz58wEAVlZWAIA5c+Ygd+7c2Lp1a1o2kYiIiIiIiIiI3mGR1g0gIiIiIiIi+ruePXuGefPm4eHDh3j58iWGDh2KpUuXokePHsiQIQOnciAiIiIiIiIiSmcUMdS/JCIiIiIiIjIDq1evRuvWrQEADRs2xKZNm+Dh4YFZs2ahUaNGads4IiIiIiIiIiIywVACERERERERmQXD7auiKDh48CCqVasGCwsL2NnZYePGjfjqq68gIhARaDScrZCIiIiIiIiIKD3gKA0RERERERGZBUVR1L9jY2MBAElJSXj9+jXOnj2rbsPsPRERERERERFR+sFQAhEREREREZkNRVEQGxuLU6dOoXz58ujTpw8AoH///hg7diwAQKvVQqfTpWUziYiIiIiIiIjo/3H6BiIiIiIiIjI7r169wsuXL+Hj44MNGzagSZMmAICxY8di6NChAACdTgetVgu9Xs/pHIiIiIiIiIiI0ghHZYiIiIiIiCjdel+OPlOmTMiVKxcAoHHjxli7di0AYPjw4Rg/fjyAtxUT4uPj1UDC77//ngotJiIiIiIiIiIiYwwlEBERERERUbokIlAUBQBw7NgxBAcHY/LkyTh+/DgiIiKg1WqRkJAAAGjSpEmKwQRra2sAb6d3KFq0KLZs2ZIGR0JERERERERE9Pni9A1ERERERESUrq1cuRKdO3dGUlISEhMTkTlzZpQuXRrz58+Hh4cHEhISYGVlBQD49ddf0axZMwCAn58fWrZsifnz52PevHlwcHDA77//Dm9v77Q8HCIiIiIiIiKizwpDCURERERERJRuhYSEwNfXFwDQrl07vHz5EtevX8f169eRK1cu7N+/H97e3ibBhA0bNqBp06YQEWi1Wuh0OuTOnRv79u2Dt7c3dDodtFptWh4WEREREREREdFng6EEIiIiIiIiSjf0ej00Gg30ej0AoHHjxjh48CAWLVqEpk2bIi4uDo8fP8ZPP/2EAwcOwMvLC0eOHEkWTNi/fz8WLlyIyMhIeHt7Y9SoUXB3d2cggYiIiIiIiIgolTGUQEREREREROnOmTNnULx4cRQtWhR16tTB1KlTAfwZWhAR1KlTB7t3735vMCE6Ohp2dnZISEiAtbU1AwlERERERERERGlAk9YNICIiIiIiIjIWEBCA0qVLo1WrVgCASpUqAQB0Oh00Gg2SkpKgKAp27tyJmjVr4sGDB6hUqRLu378PKysrJCYmAgDs7OygKAqsra0BgIEEIiIiIiIiIqI0wFACERERERERpSuOjo5QFAXr1q3DtWvXcOfOHQCAodCfhYUFkpKSACBZMOHBgwewtLSETqeDoihpdgxERERERERERPQWQwlERERERESUZgxBA+P///bbb7Fp0ybY2dlBr9dj9+7dAN6GEXQ6nfp3SsGE/Pnz49GjR6yKQERERERERESUTjCUQERERERERGlCRNRqBi9evDBZXq9ePaxZswb29vbYuXMnunXrBuDtFAzvCyaUKVMGsbGxrJBARERERERERJSOMJRAREREREREacIQHpg5cyby5cuH48ePq8veDSbMmzcP/fr1A/D+YMKJEyfw7NkzZMuWTV1PRERERERERERpi6EEIiIiIiIiSjNJSUk4dOgQIiIi0LZtW5w8edIkmFC3bl2sXr0adnZ2mD59+nuDCYa/3dzcoNfrOX0DEREREREREVE6wVACERERERERpRkLCwssX74c3333HW7fvo1mzZolCyYYKib8VTDBOISg0fBWl4iIiMyHXq9P6yYQERERfVSKiEhaN4KIiIiIiIg+b1FRUejSpQsCAwPh5eWFtWvXomzZsjDcsiqKgq1bt6JFixaIiYlB//79MXny5DRuNREREdE/JyJqADMler2eIUsiIiL6pDCUQERERERERB/du4PrOp0u2RQLfzeY0LZtW0RGRmLUqFEYMWJEqh4HERER0X9h3Cc6e/YsLl++jEOHDqFw4cLw8vKCr69vGreQiIiI6MNjKIGIiIiIiIhSzfHjx1G8eHHY2tqm+Bbgu8GE4OBglCtXziSYsH79egwePBi7d+9Gjhw50uAoiIiIiP454woJq1atQv/+/fH8+XOTbVq2bImBAweicOHCyQKcREREROaKoQQiIiIiIiJKFWvWrEGrVq3QoUMH+Pv7w9bWNsWKCREREejUqRPWrVsHLy8vrF69Gl9++SVEBCICjUaDuLg42NjYICkpCRYWFml0RERERET/3OrVq9G6dWtYWVlh6NCh8PHxQVxcHPz9/XHhwgVUrlwZ/fr1Q926dTmNAxEREX0SOHJDREREREREH52IwMrKCq6urli8eDEsLS0xderUFIMJTk5OGDRoEC5duoSrV6+iZcuWasUEAxsbGwBgIIGIiIjMytmzZ9G/f39oNBosX74cLVq0UNdZWlqia9euOHz4MNq0acNAAhEREX0y2KshIiIiIiKij05RFNSvXx9Lly6Fp6cn5s2bh/79+yM2NhZarRY6nc5k+xIlSsDNzQ0A8ODBA1SrVg1nz55VSx4TERERmaPz58/jyZMnGD16tEkg4dixY5g2bRrevHkDPz8/dOjQAQDUKaxY8JiIiIjMGUMJRERERERE9NEYD6BbWlqiVq1amD179l8GE+Lj46EoCooXL44ff/wRDRs2REJCArJmzZpWh0FERET0nxj6RFu3bgUAlC9fXl134sQJdO3aFefOnYOfnx8mTJigrgsLCwPwNuDJYAIRERGZK4YSiIiIiIiI6IN5d7D83coGKQUTBgwYoAYT4uLiYG1tDeDtoL1Op0NQUBCe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HN999x08PDwwa9YsNGrUKK2bSZTqdu/ejXr16sHR0RFz585Fs2bN1HXGlRQiIiIQEhKC+fPn49SpU9BoNBAR2NraIiYmBvnz58euXbvg5eUFnU7HNwWJiIgo3TLu4xw/fhxjx459bzABAF6+fAl/f38cPHgQoaGhAIC6deuicePGsLe3x48//oivv/4aGzduTO1DISIiIkozDCUQEREREaXg2bNnmD17NsaPHw9HR0e8fv0anp6emDt3LurVqwfAdICS6FOm1+sRExODn376CcHBwVi4cCE6dOigrktp3uOEhAS8ePECkyZNwu+//47Lly+jRIkSKFGiBHr16oUsWbIwkEBERERm4a+CCaNHj0apUqUAAImJibC0tERSUhJiYmKwaNEiBAUF4dKlS0hKSkLOnDnx8uVLvH79GkuXLsX333+fhkdFRERElHoYSiAiIiIi+gt169bFzp07odFo0K1bN8yYMQPAnwOORJ+LsLAwlCpVClZWVrh8+TKsrKzeG0gATAfvk5KS8PLlS5MgAgMJREREZE7+bjDh3T7OzZs3cfbsWQwbNgwvXrxAREQEvLy8cPToUXh6eqbJsRARERGlNou0bgARERERUXq1e/du7NixA/b29oiOjsbatWtRqlQptGnTBpaWln/5QJboU/Po0SM8e/YM+fPnh4WFBUTkL89/vV6vhg8sLCzg5uYGAOpnGEggIiIic6IoihpMKF++PIYPHw4A2LFjBwCowQStVgvDe4CKoiBPnjzIkycPKlSogIMHD2Lz5s2YNGkSPD09GdIkIiKizwZHUImIiIiI3sPHxwft2rXD/PnzMW7cODx9+hT9+/fHypUrAbx9uKrX69O4lUSpw8LCAlqtFrdv38adO3feO3WJXq9HfHw8Zs+ejQcPHqgD7YYwAqc8ISJKn+7evZvWTSBKl4wLDRuCCQDUYELt2rWxY8cOjBw5EqdPn1a3e3cfHh4eaN26NVavXo0cOXIgKSmJgQQiIiL6bDCUQERERERkREQgItDr9ciZMyfmzZuH1q1bo0uXLhgyZAieP3+OgQMHJgsm6HS6NG450cdVuHBhfPXVV4iKisKyZcvw5s2bZNvodDpoNBrExMTg559/xujRo5GUlJQGrSUion9i0qRJKFu2rPrGN9HnzjiIEB8fj/DwcNy+fRsJCQkm25UvXx7Dhg17bzDBuGKC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ZgfXG264YW1JB2gvlp7x4K233oolS5a0eP03Lr2wwQYbxJVXXrnMv/WlL30pqqqKSy+9tMXfjnCySfsxZcqU2GSTTeLwww+PT37yk1FXVxeHHHJIzJ49+z1/74033ojnnnsu5syZExH/fM2bkpX24N8JB95+++1YtGhRDBkyJKqqarasSeP5xFlnnRWDBg2KU045JTbffPN47LHH0sYMH7QP8rhou+22i8985jOxePHi+PWvfx29e/euhQkvvPBC6vOAD8J1110XK620Uqy55ppx8cUX124/+uija0uy9e/fPyZPnly7b+l9SdOf995776iqKi666KKIcHxE27b067PxW9+TJk1q8Zh/FSZ4rfNh8Pjjj0e/fv1irbXWiksuuSTmzZsX1113XW05w/XWWy8ef/zxiGh+7fSMM86IqqpipZVWihNOOCEi/rlvcI0IeD9ECdBGLVq0KI488sioqioGDx4c8+fPj+uvvz6qqor6+voYP358i9+ZP39+dOvWLYYMGVK77dVXX639u3H6yTfeeEPlTrs3c+bM6Nq1a+ywww61297rpLHpwfX5558fJ5xwQvzsZz+rHXxDe9H0IuFNN90Uhx12WGyxxRaxxRZbxP7779/sm0+zZs2qzRCy4oorxllnnRVPPfVUvPbaa7F48eL47ne/G1VVxdChQ2sz8EB71Lhd/P3vf4+Id5b22WSTTaKuri4OPfTQFmHCssKDxtt8O5z2oulr9YEHHoi77rorJk+eHIsXL17ma3yfffaJqqrimmuuiYh3LsY3+sxnPhObbbZZRPwzVFh6HVloi/7T46KVVlopfvOb38Rzzz0Xb775Zrz++uu1tZIPPPDAiHhnn9IYJqy11lpx7LHHOl6i1Z199tlxww03LPO+efPmxdChQ6Njx45x4YUX1m5vnClntdVWi9133z2qqor+/fs3i9TeLUxunIlq1113TXg2kOPCCy+Mq666Kg4//PDYYostarc3vs4b//ffmTEB2rM77rgjqqpqNlNIRMQrr7wSO+2003uGCaeddlrtC13jxo2r3S7YAd4PUQK0YbNmzYrddtut9m3uqqqiZ8+e8bvf/a72mKYnjk888UT07Nmz2bSsjUaNGhVVVcV55523vIYPqcaPHx9VVcXIkSMjwoVzPhqavuf/93//d3Ts2LG2JEnXrl1rJ4wnnnhi7cPZ+++/P771rW/V7uvfv38MHjw4+vfvX9u/NE7Z5+SS9uC9ooGms0hdccUVsemmm7YIE5r+/pQpU2qvf2ivLrnkkujZs2esvvrqUVVVfO5zn4szzzwzFi1aFBH/PEZq/FBqnXXWiZkzZ8Zrr70WEf9cuuGggw6KN998076AduODOC4aNGhQfOYzn4mNN964dlzUODVxRMTLL78cZ511Vqy88srRt29fyzjQqqZMmVKbfnvq1Kkt7r/mmmuiqqrat1ojIn75y19Ghw4dYrXVVov77rsv/vGPf9RmShs4cGBcffXVtcc23aYao4T58+fHaqutFsOGDUt8ZvDBufvuu2vv8Ztuuml84QtfWObj3i1MGD16tC+v0C4t6zz5xhtvjEGDBtXOC5YsWdJspoN/FSY0zphQVVWzZXOdLwD/KVECtGENDQ3x8ssvxxZbbBEdOnSIFVZYIU488cTa/cv6EPaSSy6Jqqqirq4uDj/88Ljgggtq34raaKONFL98aDRGCdtvv/2/DBIWL14ct956a7zyyivNbvdtWNqrK6+8snZBcty4cbFw4cJ49tlnY/z48dG5c+eoqir222+/2mw5r7zySowZMyYGDBgQPXr0iKqq4uMf/3iMGDHCmoC0K03ft6dNmxannXZanHDCCXHuuecu8z196TDh4Ycfrt131FFHxRprrBE///nPXVSh3briiitqFwp32mmn6NOnT6yyyirRuXPnOPTQQ+Pll19u9vjGD6Lq6+tjiy22iE984hO1YG3BggWt9Czg/flPj4v69+8f3bt3j6qqolevXrHjjjvGk08+GRHvHBc17ldeeumlGDt2bDzyyCOt9hyh0f777x9VVUWPHj3i5ptvbnbfxIkTY//9968tNTJx4sRYc801o3PnznHXXXdFxDvXkSZMmBCdOnWKTp06xcCBA5st5bC0xi+47L777i2WEIW26rDDDqsdHw0cOPBd37+XDhM22mijqKrK+QHtTtNz4XvvvTeuueaamDhxYvziF7+Ibt26tVjCuel11P9LmDB27NjkZwJ82IkSoI274YYboqqqWGGFFWpTbN9zzz3v+vhXX301fvnLX9YOFhr/6927d8yaNWs5jhxyzZ49O3r37h39+/evrQO+dJzQ+CHr/fffH8OGDXvXtWOhPXnyySdr64I3nTkn4p0Zdhpn1jnqqKNa/O4TTzwRc+bMiZtvvjkWLFhQuzgvSKC9ueiii6JDhw7NjnW23377uPXWW5tNSR/RPEzYY489YsKECXHggQfWvk3b+AEUtCcNDQ2xaNGi2GWXXaJbt24xYcKEiIh49NFHY8yYMbH++utHVVXxzW9+s0WYsPfee8daa61V2waGDh3a7INYaE/ez3HR448/HjNnzoxrr702Zs2aVdtWmm4HS0/1Da2l6blu42wfywoT5s2bV3u97rffftGpU6faPqLp39hoo41qx1Jdu3aNm266qcX/59SpU6NHjx7RuXPnFkthQVvU9P27canCFVdcMc4555x3/Z2mYcJvfvOb2GqrrcykRrt1ySWXxKqrrlo7R15nnXVijTXWiIsuuqhFaLOsMKFv374xb968Fn+3aZhw/vnnpz8P4MNLlABtzNIXO6ZPnx7Dhg2LU089Nfbcc8+oqioGDx4cd99993v+nZtvvjkOOOCA2HfffeOEE06I+fPnJ44aPnj/6sLfwoULY9ttt619a6NR40F204Prr371q83WUIb27MEHH4zOnTvH3nvv3ez2GTNmxCc/+cmoqiqOPfbYZf7usj5scpGd9ua6666rXRA55JBD4kc/+lF87GMfi6qqYuONN45JkybVgptGV199dWyzzTbNIoYBAwbUvgnig1jaoxdffDF69OgRxx9/fLPbFy9eHL///e+jb9++7xom/PWvf43rr78+7r///tq09rYD2iPHRXyUNH3NfvOb36yFCcsKCh599NGoq6uLvn37xlNPPVU7T26MN7fZZpvYa6+9Yv/994/u3bs3W7ak0bx582L06NExc+bMpGcEH7ymM3ocfvjhUVVVrLLKKu95Pajxvf+NN95oNs09tCfXXHNN1NXVRVVVsccee8SXvvSl2oxR2223XTz22GMtfqfptdNddtklqqqKLbbYIt5+++0Wx0Q///nPY8UVV4y//vWv6c8F+PASJUAb0nRnP3PmzDjvvPPiscceq508Pvroo7HbbrtFVVUxZMiQFmFC44FE499ZvHhxRFjnifan6bYwd+7cmDJlSpxzzjlx6aWXxmuvvRZvvvlmRETcd999sdpqq0VVVfH1r389Xn311Rav99GjR0dVVTF8+PB4/vnnl+vzgPdrWe/fl112WYsL7HfeeWdtCu6jjz662ePnzZsXt99+e/pYIcvS28FBBx0UnTt3jkmTJtVue+qpp2LkyJHRqVOnGDBgQEycOLFFmHDPPffEySefHLvsskscffTR8cwzz0SEC460D0tfFGxoaIjFixfHuuuuW1tXvOlFxbfeeutfhglNOV+gPXBcxEddQ0NDs+OWQw45JKqqim7durWYMWHu3LmxwgorRN++fWvnwY1BQkNDQ/Tq1Su+8Y1vxPPPPx8vvvhiRCz7mKjx3BvakqWPixYvXvyur9UjjzwyqqqKzp07x7XXXvtv/01o65Y+Ltpvv/1itdVWi4kTJ9Zuu++++6JPnz5RVVV8/vOfX+Z10abnEHvvvXdtNtpl+dvf/vYBjBz4KBMlQBvR9OD3hhtuiEGDBkVVVbHrrrvW1nd9++23Y/bs2fGVr3ylFiY0XRew0fjx42snlUv/bWjrmr5er7/++ujfv3+zb7Z+6lOfijFjxsTChQsjIuIPf/hDbWqy4cOHx0knnRS33nprTJ48uTa7yNprr226SdqdptvC/fffX/t50qRJUVVVjBw5MiIi7rjjjmVeeG+86Hj22WdHnz594sEHH1yOo4f/zJ/+9Kd3vW/GjBkxe/bs6NevXxx44IG12xu/DfXMM8/EYYcd9p5hQsQ7x0yNF3AECbQHTfcHf/jDH+IHP/hB7LTTTnHyySdH3759Y9y4cRHRchmrpcOEAw88MF555ZWI8Nqn/XFcxEdd0w+f/vjHP8a1114bp5xySvTu3bs2Y0JjpNaocZaoY489Nl544YXa7Y3T2p977rm121w3or1o+lq99tprY7/99ot+/frF0KFDY//994/f//73tVmgGv27YQK0Ve91nnz77bfHww8/HBtssEEcfPDBtdsbQ52HH344Nt1001qY8Nxzz7X4G8s6jwDIIEqANmbSpEnRqVOnqKoqfvjDH8acOXNaHAgsHSbceeedtftOOumkqKoqvvzlL/vGE+3a1Vdf3Wx67gkTJsTo0aOjc+fOsf7668eoUaNqF1ZmzJhRu+DeuGZg47833XRT003Srl1wwQVRVVWMGjUqIiIWLFgQffr0iU996lMxYcKE2tTEy7rw/uabb8agQYNik002MVMIbd6vfvWrqKoqTj311Bb3TZw4MaqqigMOOCCGDh0aY8aMiYh/BgmNFyffK0xwsZ327sILL2wWajb+t9tuu9Ues3RssHSYsMcee9SmJYb2yHERH0VNj2EuvPDC6NGjR+16ULdu3WKttdaKqqqiZ8+ezcKECRMmxDrrrBPdunWLXXfdNc4888za9Nyf+MQnbAe0a02Pizp27Fi7ltqrV6/YZ599WixJIkygvfp3zpP322+/2HLLLeO3v/1tRPzzPLnxs4GHH344Ntlkk/cMEwCWB1ECtCFTpkyJqqqiS5cucd55573r4xoaGmL27Nm1pRw23HDDuOyyy2pT93Xv3j3+8pe/LMeRwwfrtttui/r6+ujSpUvtgDoi4tRTT62daK6++urxgx/8oHYh5dFHH42xY8fG7rvvHjvuuGN89atfjXHjxi1zbUxoL+68887atnD22WdHRMRLL70Ue++9d+0bUUtPWfzaa69FxDv7ipEjR0ZVVTF69GhTr9Lm3XnnnbULi1OmTGl231VXXRXrrrtudOzYsRYnLO3dwoQrrriitqQVtFd33HFHdO7cObp06RK//OUv4+yzz46DDjqotk386Ec/qj12WWHCDTfcEKuvvnr07NnTtKu0W46L+Ki78soro6qqWGutteLiiy+OhoaGmD9/ftxyyy2x44471raDxqUc/v73v8evf/3r2GijjZrFbAMHDownnngiIizfQ/t02223xSqrrBI9evSIsWPHxr333hu33nprjBgxojZV/dZbbx1PPvlks99rDBO6du3abCk4aMv+L+fJ3/rWt1r8/rLChC984QvCNKBViBKgjXj++edj6623bjGFXtNZEl5++eV46qmnaj8//vjjsc8++zQ7udxwww19K5x27cUXX4yvfe1rUVVVnHnmmbXbTz311FhhhRWiU6dOccwxx0SfPn2iS5cuceSRR7YofF1Yob1a+rV7xhlnRFVVLS6Y3H///VFfXx9VVcUmm2yyzNd84wWXbbbZprbcCbR19957b3zjG99Y5n3XXXddDBo0KDp06BCbb755/PnPf27xmKXDhM6dO0ePHj3iuuuuyxw2fOCWfl8/88wzo6qqZmvE/v3vf48LLrggOnToEFVVxfHHH1+7b+kw4c0334ypU6fWYk3HSrQHjovgHQ0NDfHqq6/GDjvsEFVVxcUXX7zMx+27775RVVXU19fHTTfdFBERixcvjocffjiOPvroOOqoo+K0006rfRBlKR/ai6Xf10855ZQWx0UR76x3P3HixNhss82iqqr40pe+1OI9/6ijjoqqqqJfv361cA3aun/3PHnw4MFx3333tXjMssKET3/6082W9gFYHqqIiAK0uvnz55fNNtusfOITnyjTpk2r3f7666+Xp59+uhxzzDFl9uzZZe7cueU73/lO+cpXvlI233zzUkopp556apk9e3bp1q1b+fa3v13WX3/91noa8L49+OCDZfjw4WX33Xcv48aNK6WUcu6555YjjjiivPHGG+W2224rQ4cOLaNHjy6nnXZaWWONNcqee+5ZjjnmmNK9e/fS0NBQ6urqSimlRESpqqo1nw78R8aMGVN69OhRHn/88XLdddeV6dOnl1JKefvtt0tdXV2pqqrcdtttZeeddy6vvfZa2W677cqwYcPK1ltvXV544YVy/vnnlxtvvLH06dOnTJ8+vayzzjrNtg1oD37zm9+UV199tRx11FG126699toyatSo8sgjj5T99tuvnHjiiaV3797Nfq/xvf+5554ro0aNKtOnTy+333576dWr1/J+CvC+Ne4Pnn322TJ58uQyZcqUFsc3l112Wdlnn31KQ0NDOe6448oJJ5xQSnlnn9GhQ4cWf/Pdboe2ynERlPK3v/2tDBgwoNTV1ZWZM2c2O/ddsmRJ6dixYymllF133bVMnjy5dO/evVx22WXlc5/73DL/nn0B7dGYMWNKfX19efzxx8ukSZPK3XffXUpp/npesmRJufnmm8v3vve98thjj5WTTz65fPe73y1vvfVW6dSpUymllJ/85CdlxIgRpU+fPq31VOA/9m7nyUcddVSZPXt22X///cuPf/zjFufJjfuM2bNnl+HDh5dnn322PPvss6W+vn55PwXgI0yUAG3Egw8+WD75yU+Wbbfdtvz+978vK6+8cpk3b14ZP358ueCCC8oTTzxR6uvry8KFC0uHDh3K17/+9fKrX/2q9OzZs7WHDh+oZ555pvzsZz8rhxxySBk4cGCZNm1aOfjgg8ucOXPK5MmTy0477VRKKWXevHll5513LnPnzi1rrbVW2XfffcsPf/jD0r1791Z+BvD+PPvss2XttdcupZTSr1+/svbaa5dbbrml2WMaTybvueee8p3vfKc89NBDZfHixaVjx45lyZIlZYUVVijDhw8v559/fundu7eLjrQ7CxcuLGuuuWYppZQzzjijfPe7363dN3ny5HLEEUeUuXPnlm9/+9tl9OjRtW2mUeOHtgsXLiwdO3Ysa6yxhu2Adqfp/mDQoEGld+/e5YYbbljmY/+vYQK0F46L4B2vvPJK2Xjjjcvbb79dHnjggdK9e/dmkVrj6/rpp58uO++8c3nooYdKt27dysSJE8t2221Xe6xwn/aq6f5gwIABZbXVVqtFCUtbtGhROfvss8vRRx9ddthhh/K///u/pZTSLOBZ1s/Q1r3f8+TGY6Z58+aVlVdeufTq1UuoCSxX3m2gjejbt2/ZaqutyrRp08qRRx5ZjjvuuLLddtuVE044oXTv3r389Kc/LTNnziyXXnpp6dWrVxk/fnyZNWtWaw8bPnC9evUqJ598cvn4xz9eSinl9ttvL4888kj5yU9+UnbaaafS0NBQ3n777dK3b9/yta99rXTq1KmsuOKK5ZRTTilnnnlmaWhoaOVnAO/PWmutVaZOnVpKKWXOnDnltddeK48//niJd5bdKqWUUldXVxoaGsqQIUPKpEmTyoQJE8oBBxxQRowYUQ4//PBy1VVXlcsvv9yFd9qt+vr6ctddd5W6urry/e9/v5xxxhm1+774xS+W008/vXzsYx8rY8eOLT/96U/LggULmv1+40X3+vr6ssYaa5SIsB3Q7jTdHzz00EPl5ZdfLk8//XSz/UGjPffcs4wfP77U1dWVE088sZx44omllOJ1T7vnuAjeUVdXVzp37lwWLFhQfvvb35ZS/nm8U8o77/cRUXr06FHWXnvtUlVVefHFF8v2229f7rjjjlqIIEigvWq6P3j44YfLwoULy5///OcWx0SllLLqqquWPfbYo6y++urlxhtvLH/5y19KKaVFgCBIoL15v+fJjcdMffv2Lb169arNOgWw3CzPtSKA9zZr1qzYeOONo6qqqKoq6urq4uCDD4758+fH66+/XnvciBEjoqqquOiii1pxtJDv5ZdfjiFDhkSHDh3itttui4h31tN86623IiLi2GOPjbXXXjvGjBkTm266acyaNas1hwsfqFtuuaW2PzjrrLNqtzc0NPzbf8Oa4bR399xzT207OP3005vdN3ny5OjXr19UVRUHH3xwPP30060zSEjWdH9wzjnn1G5f1v5gwoQJsdJKK0VVVXHqqacuz2FCKsdFEHHeeedFp06dYujQoTF16tTa7Y3bwZtvvhkREaNGjYrPfvaz8eUvfzk6deoUTz31VKuMFzI03R+cdNJJy3zMG2+8ERERW265ZVRVFXfdddfyHCKkc54MtFeWb4A25rnnniu33XZbWbRoUenbt28ZNmxYKaU0m2Jv2LBhZc6cOWXGjBllww03bM3hQqrXXnutbLvttuXee+8t48ePL3vttVez+7fffvvy0ksvlWnTppUlS5aU1VdfvZVGCjmmT59ePvvZz5ZSSrn88svLV7/61VJKaTHt6r/6Gdqze++9t2yxxRallFJOO+208v3vf7923/XXX18OP/zwMnfu3HLooYeWo446qqy77rqtNFLI03R/cNlll5Wvfe1rpZRlv99feOGF5YQTTijTpk0r66+//vIeKqRxXMRH3cKFC8uee+5ZbrnllrLHHnuU73//+2XLLbcspZTyxhtvlBVXXLGUUsomm2xSNt544zJhwoTyj3/8o3Tt2tUsIXyoNN0fXHLJJWXvvfeu3de4LUREGThwYFm0aFG56667Su/evVtptJDDeTLQHpmbBdqYNddcs+y+++5l5MiRtSDhjTfeqF1EOfroo8uMGTPKpz/96dKzZ8/WHCqkW3nllctee+1VVlhhhXLppZeWu+66q3bf6NGjy9SpU8vmm29eVl55ZUECH0rDhg0r06ZNK6WUsscee5RJkyaVUppP1dr4c1MuvPNhMnjw4PLHP/6xlFLKEUcc0WyKyl122aWcfvrpZcCAAeXss88u55xzjmV8+FBquj/Yc88933V/UEopI0eOLLNmzSrrr79+WbJkyXIfK2RxXMRHXX19fRkzZkzp27dvufzyy8t//dd/lfHjx5dSSi1IOOKII8pDDz1UNthgg1JKKV26dLGMFR86w4YNK7feemsppZQRI0aUs846qzz//POllH9uC4cffnh55JFHysCBA0vXrl1baaSQx3ky0B6ZKQHasKW/0XHkkUeW008/vay//vpl6tSptZNM+DB75plnyje+8Y1y0003lY9//ONl4MCB5aWXXipTp04tvXv3LtOmTSt9+/Zt7WFCqttuu61su+22pZT3/mYgfJi91zdBrrzyynLGGWeU8ePHl/XWW6+VRgj57A/AdgCzZs0qI0aMKPfff39paGgoW221VenWrVtZsGBB+dOf/lT69etXpk+fXtZcc83WHiqkaro/2Hnnncv6669fmyXkjjvuKBtuuGGZNm1aWXvtte0j+NByngy0J6IEaOOeeuqpMn369DJ27NgyY8aM0q9fv3L11VeXjTbaqLWHBsvNY489Vn70ox+VyZMnl1dffbV06NChDBgw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UQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAAAApBAlAAAAAAAAAAApRAkAAAAAAAAAQApRAgAAAAAAAACQQpQAAAAAAAAAAKQQJQAAAAAAAAAAKUQJAAAAAAAAAEAKUQIAAAAAAAAAkEKUAAAAAAAAAACkECUAAAAAAAAAAClECQAAAAAAAABAClECAAAAAAAAAJBClAAAAAAAAADw/9q786Cq6v+P4y8EBaFBQhYdUbxYJrmgkDqJjjg6miWKmiCaQlnmRrlhM4450+6W1lASjYI6QykCQaFN2hgUiQvimAa4cnEXyVRcWAx+fzjciR+LgFywr8/HDDPn3vfnfD7vc/n8d1/3HABmQSgBAAAAAAAAAAAAAACYRaNCCQUFBUpJSdHy5cs1evRoOTk5ycLCQhYWFgoNDa3XHDk5Ofriiy8UEhIib29vubm5ycbGRnZ2dvLw8FBQUJCSk5NVUVFRr/l27dql4OBgeXh4yNbWVjY2NurcubMCAgIUFxdX5zxGo9HU/4P+6nt9/3bp0iU5ODiY5vDz86tz/PHjx7Vu3ToFBATIYDCobdu2srW1lcFgUFBQkHbs2FHvzwUAAAAAAAAAAAAAgJZi1ZiTXF1dH3rhjz76SLGxsTXW8vLylJeXp7i4OA0dOlSJiYlydHSscWxpaammTZumuLi4arXz58/r/PnzSk5OVmRkpJKSktSuXbuH7r2hwsLCdOPGjXqNDQkJ0ZYtW2qsGY1GGY1GxcXFadSoUdq6dascHByasFMAAAAAAAAAAAAAAJpOo0IJ/9a5c2d5enpq165dDVvYykoDBw6Ur6+vevfurQ4dOsjZ2Vl///23cnNzFRUVpWPHjiktLU3+/v767bff1KpV9Rs7zJ8/3xRIcHFx0ZIlS+Tt7a3WrVvr6NGjWrlypfLz85WamqopU6Zox44ddfb14Ycfaty4cbXWn3zyyQZd5w8//KCEhAS5uLiooKDggeMvXLggSXJ0dNTLL78sPz8/de3aVVZWVjp8+LDWrl2r48eP66effpK/v7/S0tJq/FwAAAAAAAAAAAAAoKHKy8u1fv16RUdHKzc3V1ZWVurXr58WLVqksWPHtnR7aAZNvQcaFUpYvny5+vfvr/79+8vV1VVGo1EGg6FBc2zYsEFWVjUvP2LECM2ePVuBgYFKTEzU3r17tWPHDvn7+1cZV1BQoKioKEn3wwKHDh2Sm5ubqT548GBNnTpVXl5eMhqN2rlzp7KysuTt7V1rX506dVKvXr0adC21uXXrlubOnStJWrNmjaZPn/7Ac9zc3BQVFaWQkBBZW1tXqfXv31+vvPKKRo0apfT0dKWnpys2NlbTpk1rkn4BAAAAAAAAAAAAPL4qKioUGBiohIQEdevWTTNmzFBJSYmSk5M1btw4RUREaN68eS3dJszIHHugUT+xf++99zRmzJiHeoxDbYGESpaWllqyZInp9a+//lptzL59+1ReXi5JevXVV6sEEirZ29trwYIFptd79+5tbMsNtnTpUp07d07Dhg2rd3Bg06ZNmjlzZrVAQiVbW1tFRkaaXsfHxzdJrwAAAAAAAAAAAAAebwkJCUpISJCvr6+OHj2qiIgIff311/rzzz/l7u6uxYsXy2g0tnSbMCNz7IFH+r7/dnZ2puPi4uJq9dLSUtOxh4dHrfN069bNdFxSUtJE3dXtwIED+vLLL9WmTZsqIYKm0KtXLzk5OUmSTp8+3aRzAwAAAAAAAAAAAHg8JSUlSbr/4+u2bdua3ndyctKCBQtUUlKimJiYFuoOzcEce+CRDiV8++23puMePXpUq3fv3t10fObMmVrn+fcX9/8+x1zu3bunmTNnqry8XO+8846eeeaZJl+jMpDRqtUj/S8EAAAAAAAAAAAA8B9x5coVSZLBYKhWq3xvz549zdpTSzp58qSk+99Fz507V5mZmS3ckfmZYw88ct9oFxYWKiMjQzNmzNAnn3wiSWrfvr2mTp1abWyfPn30/PPPS7r/2IOLFy9WG1NUVKTPPvtMktS1a1eNHDmyzvUjIiJkMBhkbW2tdu3aqWfPnpo1a5aysrLqfQ1r1qzRkSNH1K1bNy1durTe59XX4cOHdfPmTUk1hzUAAAAAAAAAAAAAoKGcnZ0lSXl5edVqle+dOHGiWXtqKatWrVJ0dLSk+z9Kz87OVnh4uFavXt3CnZmXOfbAIxFK8PPzk4WFhSwsLOTs7KxBgwYpOjpaFRUVcnR0VGJiohwcHGo8Nzo6Wu7u7rp27Zq8vb21du1apaamKj09XV999ZW8vLyUl5en9u3bKzY2VtbW1nX2kpWVJaPRqNLSUt28eVPZ2dmKioqSj4+PZs2a9cDHP5w5c0bvv/++JGn9+vWysbFp1GdSl48//th0HBgY2OTzAwAAAAAAAAAAAHj8jB49WpK0YsUKFRcXm97/66+/TD8Ev379egt01rwyMzP1448/1ljbuXOnDh061MwdNR9z7AGrpmrOHMLCwrRs2TK5uLjUOqZHjx7KzMzU+vXrtXr1ai1atKhKvXXr1lq0aJHeeustdenSpdZ5HBwcNH78ePn5+enpp5+WjY2NLl26pF27dmnjxo26deuWoqKiVFRUpNjY2FrnefPNN3X37l0FBQU98K4MjZGQkKD4+HhJko+PjyZOnNjkawAAAAAAAAAAAAB4/AQHBysmJka//PKLevfurRdeeEFlZWVKSkqSq6urJMnS0rKFuzS/mJiYOuvR0dHy8fFppm6alzn2gEVFRUXFwzZmNBpNz48ICQnRpk2bGnR+Xl6ebt++rYqKCl2/fl2ZmZmKjIzU6dOn9eKLL2rDhg2mC6zJli1btHLlSmVnZ9dYd3d3V1hYmBYuXCgLC4tq9dLSUt27d0+2trY1nn/y5EmNGDFCZ8+elSQlJydr7NixNfYREhIie3t75ebmqmPHjlXqlWsPHTpUqamptV5PbXJzczVgwAAVFRWpbdu2yszM1LPPPtvgeQAAAAAAAAAAAACgJiUlJVqxYoW++eYbGY1GtWvXTuPHj9fixYvVvXt3denSRfn5+S3dplkFBQWpoKCg1rqLi4u2bdvWjB01r6beA49EKKEmxcXFmjRpklJSUtS5c2ft3btXbm5u1cYtXrxYn376qSQpICBA4eHh8vLykqWlpXJychQREWFKskyaNElbt25Vq1YNf2pFenq6hgwZIkkaMWKEdu/eXaVeWFgoT09PFRYWKiIiQvPmzas2x8OEEi5evChfX18ZjUZZWFgoNjZWwcHBDb4OAAAAAAAAAAAAAGio1NRUDRs2TBMmTFBCQkJLt4MW0Ng90PBv55uJjY2NYmJiZGtrq3PnzmnJkiXVxqSkpJgCCaGhofruu+80aNAg2dnZycbGRv369VN0dLTeffddSdL27dsVGRnZqH4GDx6snj17SrofUCgvL69SX7hwoQoLC/Xcc89pzpw5jVqjNteuXdPIkSNlNBolSZ9//jmBBAAAAAAAAAAAAADNpvIR95MnT27hTtBSGrsHHtk7JVQaOXKkdu/eLVtbW924cUNWVlam2vjx45WUlCRJOn/+vDp16lTjHMXFxXJ2dtatW7fUt29fHT58uFG9BAYGavv27ZKkgoICOTs7S7p/F4PKtZcsWaJ+/frVeH5lkMDT01PLly+XJBkMBg0cOLDWNYuKijR8+HAdPHhQkvTBBx9o2bJljeofAAAAAAAAAAAAAOpy8+ZN2dvbV3kvPj5eQUFB8vHxUUZGhiwtLVuoOzSHpt4DVg8e0rIqv/i/c+eOrl69qo4dO5pqOTk5kiRXV9daAwnS/bsu9OzZU/v371dubm6je6ktv1FaWmo6XrVq1QPnycnJMQUUQkJCag0l3L17V/7+/qZAQnh4OIEEAAAAAAAAAAAAAGYzcOBAde7cWZ6enrKxsdGBAweUmpoqDw8Pbd++nUDCY6Cp98AjH0q4cOGC6fiJJ56oUqu8a8K9e/ceOE9ZWVmVcxojOztbkmRtba327ds3ep76KCsr08SJE5WWliZJmjVrVr0CDwAAAAAAAAAAAADQWEFBQUpMTNS+fftUVlYmg8GgZcuWKTw8vNqv5/G/qan3wCP9+IYLFy7Iw8NDpaWlcnd3l9ForFL39/dXSkqKpPuBAU9PzxrnuXbtmjp27KjS0lL17t1bf/zxR4N7SU9P15AhQyRJw4cP188//9zgOSwsLCRJQ4cOVWpqaq3j/vnnH02ePFnx8fGSpGnTpmnz5s2m8wEAAAAAAAAAAAAA+C9o1RKLnjhxQnv27KlzzI0bNxQcHGx6NMK0adOqjfH39zcdz58/v8pjFCqVl5fr7bffNtXGjBlTbUxSUlKtj2aQpFOnTmnq1Kmm13PmzKmz94dRUVGhN954wxRImDhxomJiYggkAAAAAAAAAAAAAAD+cxp1p4T09HSdOnXK9LqwsFDh4eGSJF9fX73++utVxoeGhlZ5nZqaqmHDhsnLy0sBAQHy8fFRhw4dZGVlpcuXL+v333/Xxo0bdfnyZUlSr169tG/fPtnZ2VWZp7S0VF5eXsrNzZUk9e7dW2FhYfLy8pKlpaWys7MVGRmpjIwMSZKrq6uOHTsmJyenqh+ChYWeeuopTZgwQQMGDJCbm5usra118eJF7dq1Sxs2bNDt27clSYGBgdq2bVtDPzLTOlLdd0pYtGiR1q5da7ruzZs3q02bNnXO26tXr0b1AwAAAAAAAAAAAACAOTUqlBAaGqrNmzfXe/z/X6IylFAfL730kmJiYuTs7FxjPT8/X+PGjdORI0fqnMdgMCgxMVF9+/atVqvvXQhmz56tdevWydraul7ja1unrlBC165dlZ+f36B5m+AJHAAAAAAAAAAAAAAANDmrlljU19dXaWlp2rNnj9LT03X27FlduXJFd+7ckb29vQwGgwYOHKgpU6bI19e3zrnc3d118OBBbd26VfHx8crKytLVq1dVUVEhR0dH9enTRwEBAZo+fXq1Oy1U+v7775WRkaH9+/crPz9fhYWFun37tuzt7eXh4aEhQ4botdde444EAAAAAAAAAAAAAAA0QKPulAAAAAAAAAAAAAAAAPAgrVq6AQAAAAAAAAAAAAAA8L+JUAIAAAAAAAAAAAAAADALQgkAAAAAAAAAAAAAAMAsCCUAAAAAAAAAAAAAAACzIJQAAAAAAAAAAAAAAADMglACAAAAAAAAAAAAAAAwC0IJAAAAAAAAAAAAAADALAglAAAAAAAAAAAAAAAAsyCUAAAAAAAAAAAAAAAAzIJQAgAAAAAAAAAAAAAAMAtCCQAAAAAAAAAAAAAAwCwIJQAAAAAAAAAAAAAAALMglAAAAAAAAAAAAAAAAMyCUAIAAAAAAAAAAAAAADALQgkAAAAAAAAAAAAAAMAs/g/xseUo215mUgAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 2500x1000 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# drop records with empty values\n",
    "dataset = dataset.dropna()\n",
    "msno.matrix(dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "13e48530",
   "metadata": {
    "_cell_guid": "87d6105e-031a-4700-ad3c-aeb664c3ea78",
    "_uuid": "996e4815-8361-4f34-b88f-4ac992195d48",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:47.745247Z",
     "iopub.status.busy": "2023-07-04T15:07:47.744248Z",
     "iopub.status.idle": "2023-07-04T15:07:47.888709Z",
     "shell.execute_reply": "2023-07-04T15:07:47.887404Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.166031,
     "end_time": "2023-07-04T15:07:47.891658",
     "exception": false,
     "start_time": "2023-07-04T15:07:47.725627",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "state                 0\n",
       "active                0\n",
       "reopen_count          0\n",
       "interactions_count    0\n",
       "made_sla              0\n",
       "requester_id          0\n",
       "days_to_resolve       0\n",
       "priority              0\n",
       "impact                0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d655cf97",
   "metadata": {
    "_cell_guid": "8c8746b2-69aa-4384-aedb-b41e2daa0975",
    "_uuid": "c03cf878-5f27-4864-ba4b-e206643c0c81",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:47.929729Z",
     "iopub.status.busy": "2023-07-04T15:07:47.928810Z",
     "iopub.status.idle": "2023-07-04T15:07:48.010294Z",
     "shell.execute_reply": "2023-07-04T15:07:48.008920Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.104514,
     "end_time": "2023-07-04T15:07:48.013476",
     "exception": false,
     "start_time": "2023-07-04T15:07:47.908962",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(118937, 9)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# drop duplicate records\n",
    "dataset = dataset.drop_duplicates()\n",
    "dataset.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "8da33e97",
   "metadata": {
    "_cell_guid": "1dc5ec35-f3e9-4eaa-9c82-e4f0d100bec0",
    "_uuid": "4a805625-d55e-4c10-8892-c4f4ccbe51a0",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:48.050043Z",
     "iopub.status.busy": "2023-07-04T15:07:48.049595Z",
     "iopub.status.idle": "2023-07-04T15:07:48.079059Z",
     "shell.execute_reply": "2023-07-04T15:07:48.077690Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.050938,
     "end_time": "2023-07-04T15:07:48.081827",
     "exception": false,
     "start_time": "2023-07-04T15:07:48.030889",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>state</th>\n",
       "      <th>active</th>\n",
       "      <th>reopen_count</th>\n",
       "      <th>interactions_count</th>\n",
       "      <th>made_sla</th>\n",
       "      <th>requester_id</th>\n",
       "      <th>days_to_resolve</th>\n",
       "      <th>priority</th>\n",
       "      <th>impact</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Open</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>True</td>\n",
       "      <td>2403.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>2 - Medium</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Closed</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>True</td>\n",
       "      <td>2403.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>2 - Medium</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Closed</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>True</td>\n",
       "      <td>2403.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>2 - Medium</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Closed</td>\n",
       "      <td>False</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>True</td>\n",
       "      <td>2403.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>2 - Medium</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Open</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>True</td>\n",
       "      <td>2403.0</td>\n",
       "      <td>57.0</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>2 - Medium</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    state  active  reopen_count  interactions_count  made_sla  requester_id  \\\n",
       "0    Open    True             0                   0      True        2403.0   \n",
       "1  Closed    True             0                   2      True        2403.0   \n",
       "2  Closed    True             0                   3      True        2403.0   \n",
       "3  Closed   False             0                   4      True        2403.0   \n",
       "4    Open    True             0                   0      True        2403.0   \n",
       "\n",
       "   days_to_resolve      priority      impact  \n",
       "0              0.0  3 - Moderate  2 - Medium  \n",
       "1              0.0  3 - Moderate  2 - Medium  \n",
       "2              0.0  3 - Moderate  2 - Medium  \n",
       "3              0.0  3 - Moderate  2 - Medium  \n",
       "4             57.0  3 - Moderate  2 - Medium  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# explore the data\n",
    "dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "524907b6",
   "metadata": {
    "_cell_guid": "d9ffd0f6-0479-485d-94e0-ec8695b19b41",
    "_uuid": "25839bec-17ce-4d1a-8498-6fd080ab71ae",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:48.119417Z",
     "iopub.status.busy": "2023-07-04T15:07:48.118969Z",
     "iopub.status.idle": "2023-07-04T15:07:48.158567Z",
     "shell.execute_reply": "2023-07-04T15:07:48.157077Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.061637,
     "end_time": "2023-07-04T15:07:48.161292",
     "exception": false,
     "start_time": "2023-07-04T15:07:48.099655",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "state: ['Open' 'Closed']\n",
      "impact: ['2 - Medium' '1 - High' '3 - Low']\n",
      "priority: ['3 - Moderate' '2 - High' '4 - Low' '1 - Critical']\n"
     ]
    }
   ],
   "source": [
    "print('state:',dataset.state.unique())\n",
    "print('impact:',dataset.impact.unique())\n",
    "print('priority:',dataset.priority.unique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "9fd567db",
   "metadata": {
    "_cell_guid": "db94b229-39bf-4211-9044-814c9b96be86",
    "_uuid": "90925d51-a565-486d-b3c6-f8c76bd7101a",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:48.200833Z",
     "iopub.status.busy": "2023-07-04T15:07:48.199605Z",
     "iopub.status.idle": "2023-07-04T15:07:48.224118Z",
     "shell.execute_reply": "2023-07-04T15:07:48.222748Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.047724,
     "end_time": "2023-07-04T15:07:48.226974",
     "exception": false,
     "start_time": "2023-07-04T15:07:48.179250",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>state</th>\n",
       "      <th>active</th>\n",
       "      <th>reopen_count</th>\n",
       "      <th>interactions_count</th>\n",
       "      <th>made_sla</th>\n",
       "      <th>requester_id</th>\n",
       "      <th>days_to_resolve</th>\n",
       "      <th>priority</th>\n",
       "      <th>impact</th>\n",
       "      <th>first_response_resolved</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Open</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>True</td>\n",
       "      <td>2403.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>2 - Medium</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Closed</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>True</td>\n",
       "      <td>2403.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>2 - Medium</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Closed</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>True</td>\n",
       "      <td>2403.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>2 - Medium</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Closed</td>\n",
       "      <td>False</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>True</td>\n",
       "      <td>2403.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>2 - Medium</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Open</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>True</td>\n",
       "      <td>2403.0</td>\n",
       "      <td>57.0</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>2 - Medium</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    state  active  reopen_count  interactions_count  made_sla  requester_id  \\\n",
       "0    Open    True             0                   0      True        2403.0   \n",
       "1  Closed    True             0                   2      True        2403.0   \n",
       "2  Closed    True             0                   3      True        2403.0   \n",
       "3  Closed   False             0                   4      True        2403.0   \n",
       "4    Open    True             0                   0      True        2403.0   \n",
       "\n",
       "   days_to_resolve      priority      impact  first_response_resolved  \n",
       "0              0.0  3 - Moderate  2 - Medium                     True  \n",
       "1              0.0  3 - Moderate  2 - Medium                    False  \n",
       "2              0.0  3 - Moderate  2 - Medium                    False  \n",
       "3              0.0  3 - Moderate  2 - Medium                    False  \n",
       "4             57.0  3 - Moderate  2 - Medium                     True  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# find tickets resolved by the first response\n",
    "def find_first_contact(x):\n",
    "    return x['interactions_count'] == 0\n",
    "    \n",
    "dataset = dataset.assign(first_response_resolved = lambda x: find_first_contact(x))\n",
    "dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "78efa1d4",
   "metadata": {
    "_cell_guid": "aa04c0a6-df9f-4f95-9339-1ce21c5f8e3d",
    "_uuid": "db748b32-25d1-4be9-881f-d3557c2bbcb3",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:48.266479Z",
     "iopub.status.busy": "2023-07-04T15:07:48.266065Z",
     "iopub.status.idle": "2023-07-04T15:07:48.292585Z",
     "shell.execute_reply": "2023-07-04T15:07:48.291034Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.050007,
     "end_time": "2023-07-04T15:07:48.295980",
     "exception": false,
     "start_time": "2023-07-04T15:07:48.245973",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>state</th>\n",
       "      <th>active</th>\n",
       "      <th>reopen_count</th>\n",
       "      <th>interactions_count</th>\n",
       "      <th>made_sla</th>\n",
       "      <th>requester_id</th>\n",
       "      <th>days_to_resolve</th>\n",
       "      <th>priority</th>\n",
       "      <th>impact</th>\n",
       "      <th>first_response_resolved</th>\n",
       "      <th>ticket_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Open</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>True</td>\n",
       "      <td>2403.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>2 - Medium</td>\n",
       "      <td>True</td>\n",
       "      <td>52</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Closed</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>True</td>\n",
       "      <td>2403.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>2 - Medium</td>\n",
       "      <td>False</td>\n",
       "      <td>52</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Closed</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>True</td>\n",
       "      <td>2403.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>2 - Medium</td>\n",
       "      <td>False</td>\n",
       "      <td>52</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Closed</td>\n",
       "      <td>False</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>True</td>\n",
       "      <td>2403.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>2 - Medium</td>\n",
       "      <td>False</td>\n",
       "      <td>52</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Open</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>True</td>\n",
       "      <td>2403.0</td>\n",
       "      <td>57.0</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>2 - Medium</td>\n",
       "      <td>True</td>\n",
       "      <td>52</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    state  active  reopen_count  interactions_count  made_sla  requester_id  \\\n",
       "0    Open    True             0                   0      True        2403.0   \n",
       "1  Closed    True             0                   2      True        2403.0   \n",
       "2  Closed    True             0                   3      True        2403.0   \n",
       "3  Closed   False             0                   4      True        2403.0   \n",
       "4    Open    True             0                   0      True        2403.0   \n",
       "\n",
       "   days_to_resolve      priority      impact  first_response_resolved  \\\n",
       "0              0.0  3 - Moderate  2 - Medium                     True   \n",
       "1              0.0  3 - Moderate  2 - Medium                    False   \n",
       "2              0.0  3 - Moderate  2 - Medium                    False   \n",
       "3              0.0  3 - Moderate  2 - Medium                    False   \n",
       "4             57.0  3 - Moderate  2 - Medium                     True   \n",
       "\n",
       "   ticket_count  \n",
       "0            52  \n",
       "1            52  \n",
       "2            52  \n",
       "3            52  \n",
       "4            52  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# calculate total tickets count per requester\n",
    "dataset['ticket_count'] = dataset.groupby('requester_id')['requester_id'].transform('count')\n",
    "dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "5a086a31",
   "metadata": {
    "_cell_guid": "94ab63e9-9799-444a-8942-a1cd249097b4",
    "_uuid": "b7c2824f-4b4c-4961-86f5-df7a5e44d4fb",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:48.335212Z",
     "iopub.status.busy": "2023-07-04T15:07:48.334770Z",
     "iopub.status.idle": "2023-07-04T15:07:51.091250Z",
     "shell.execute_reply": "2023-07-04T15:07:51.090137Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 2.779009,
     "end_time": "2023-07-04T15:07:51.093628",
     "exception": false,
     "start_time": "2023-07-04T15:07:48.314619",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>avg_state</th>\n",
       "      <th>avg_open</th>\n",
       "      <th>made_sla</th>\n",
       "      <th>avg_priority</th>\n",
       "      <th>total_ticket</th>\n",
       "      <th>avg_first_res</th>\n",
       "      <th>avg_interactions</th>\n",
       "      <th>avg_days_taken</th>\n",
       "      <th>avg_reopens</th>\n",
       "      <th>avg_impact</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>requester_id</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2.0</th>\n",
       "      <td>Closed</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>4 - Low</td>\n",
       "      <td>3</td>\n",
       "      <td>False</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3 - Low</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4.0</th>\n",
       "      <td>Open</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>17</td>\n",
       "      <td>False</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>44.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2 - Medium</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5.0</th>\n",
       "      <td>Open</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>13</td>\n",
       "      <td>False</td>\n",
       "      <td>3.384615</td>\n",
       "      <td>5.384615</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2 - Medium</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6.0</th>\n",
       "      <td>Open</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>16</td>\n",
       "      <td>False</td>\n",
       "      <td>5.125000</td>\n",
       "      <td>114.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2 - Medium</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7.0</th>\n",
       "      <td>Closed</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>3 - Moderate</td>\n",
       "      <td>3</td>\n",
       "      <td>False</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2 - Medium</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             avg_state  avg_open  made_sla  avg_priority  total_ticket  \\\n",
       "requester_id                                                             \n",
       "2.0             Closed      True      True       4 - Low             3   \n",
       "4.0               Open      True      True  3 - Moderate            17   \n",
       "5.0               Open      True      True  3 - Moderate            13   \n",
       "6.0               Open      True      True  3 - Moderate            16   \n",
       "7.0             Closed      True      True  3 - Moderate             3   \n",
       "\n",
       "              avg_first_res  avg_interactions  avg_days_taken  avg_reopens  \\\n",
       "requester_id                                                                 \n",
       "2.0                   False          1.000000        1.000000          0.0   \n",
       "4.0                   False          9.000000       44.000000          0.0   \n",
       "5.0                   False          3.384615        5.384615          0.0   \n",
       "6.0                   False          5.125000      114.000000          0.0   \n",
       "7.0                   False          1.000000        0.000000          0.0   \n",
       "\n",
       "              avg_impact  \n",
       "requester_id              \n",
       "2.0              3 - Low  \n",
       "4.0           2 - Medium  \n",
       "5.0           2 - Medium  \n",
       "6.0           2 - Medium  \n",
       "7.0           2 - Medium  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# group tickets by the requester\n",
    "# categorical data is aggregated using the mode\n",
    "# numerical data is aggregated using the mean\n",
    "from statistics import mean, mode\n",
    "tickets = dataset.groupby('requester_id').agg(\n",
    "    avg_state=pd.NamedAgg(column='state', aggfunc=mode),\n",
    "    avg_open=pd.NamedAgg(column='active', aggfunc=mode),\n",
    "    made_sla=pd.NamedAgg(column='made_sla', aggfunc=mode),\n",
    "    avg_priority=pd.NamedAgg(column='priority', aggfunc=mode),\n",
    "    total_ticket=pd.NamedAgg(column=\"ticket_count\", aggfunc=mean),\n",
    "    avg_first_res= pd.NamedAgg(column=\"first_response_resolved\", aggfunc=mode),\n",
    "    avg_interactions=pd.NamedAgg(column='interactions_count', aggfunc=mean),\n",
    "    avg_days_taken=pd.NamedAgg(column='days_to_resolve', aggfunc=mean),\n",
    "    avg_reopens = pd.NamedAgg(column='reopen_count', aggfunc=mean),\n",
    "    avg_impact = pd.NamedAgg(column='impact', aggfunc=mode)\n",
    ")\n",
    "tickets.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "8a285244",
   "metadata": {
    "_cell_guid": "3492069d-48ff-4c95-9dcb-2b61de7353f4",
    "_uuid": "cf08b296-4613-4b1f-a285-732adfb8ac7c",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:51.135028Z",
     "iopub.status.busy": "2023-07-04T15:07:51.134044Z",
     "iopub.status.idle": "2023-07-04T15:07:51.142318Z",
     "shell.execute_reply": "2023-07-04T15:07:51.141117Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.032062,
     "end_time": "2023-07-04T15:07:51.144873",
     "exception": false,
     "start_time": "2023-07-04T15:07:51.112811",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(5089, 10)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tickets.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "5625ad17",
   "metadata": {
    "_cell_guid": "79fe3162-3316-4750-a405-4a19974a2983",
    "_uuid": "f29bac7a-86f9-4855-8b2a-f126d0282d8c",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:51.184679Z",
     "iopub.status.busy": "2023-07-04T15:07:51.184271Z",
     "iopub.status.idle": "2023-07-04T15:07:51.215254Z",
     "shell.execute_reply": "2023-07-04T15:07:51.213970Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.054369,
     "end_time": "2023-07-04T15:07:51.218164",
     "exception": false,
     "start_time": "2023-07-04T15:07:51.163795",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>avg_state</th>\n",
       "      <th>avg_open</th>\n",
       "      <th>made_sla</th>\n",
       "      <th>avg_priority</th>\n",
       "      <th>total_ticket</th>\n",
       "      <th>avg_first_res</th>\n",
       "      <th>avg_interactions</th>\n",
       "      <th>avg_days_taken</th>\n",
       "      <th>avg_reopens</th>\n",
       "      <th>avg_impact</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>requester_id</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2.0</th>\n",
       "      <td>Closed</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>False</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4.0</th>\n",
       "      <td>Open</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>3</td>\n",
       "      <td>17</td>\n",
       "      <td>False</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>44.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5.0</th>\n",
       "      <td>Open</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>3</td>\n",
       "      <td>13</td>\n",
       "      <td>False</td>\n",
       "      <td>3.384615</td>\n",
       "      <td>5.384615</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6.0</th>\n",
       "      <td>Open</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>3</td>\n",
       "      <td>16</td>\n",
       "      <td>False</td>\n",
       "      <td>5.125000</td>\n",
       "      <td>114.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7.0</th>\n",
       "      <td>Closed</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>False</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             avg_state  avg_open  made_sla  avg_priority  total_ticket  \\\n",
       "requester_id                                                             \n",
       "2.0             Closed      True      True             4             3   \n",
       "4.0               Open      True      True             3            17   \n",
       "5.0               Open      True      True             3            13   \n",
       "6.0               Open      True      True             3            16   \n",
       "7.0             Closed      True      True             3             3   \n",
       "\n",
       "              avg_first_res  avg_interactions  avg_days_taken  avg_reopens  \\\n",
       "requester_id                                                                 \n",
       "2.0                   False          1.000000        1.000000          0.0   \n",
       "4.0                   False          9.000000       44.000000          0.0   \n",
       "5.0                   False          3.384615        5.384615          0.0   \n",
       "6.0                   False          5.125000      114.000000          0.0   \n",
       "7.0                   False          1.000000        0.000000          0.0   \n",
       "\n",
       "              avg_impact  \n",
       "requester_id              \n",
       "2.0                    2  \n",
       "4.0                    1  \n",
       "5.0                    1  \n",
       "6.0                    1  \n",
       "7.0                    1  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# encode categorical features\n",
    "cleanup_nums = {\n",
    "    \"avg_impact\": {\n",
    "        \"2 - Medium\": 1, #\"Average\"\n",
    "        '1 - High': 0, #\"Bad\"\n",
    "        '3 - Low': 2, #\"Good\"\n",
    "    },\n",
    "    'avg_priority': {\n",
    "        '3 - Moderate': 3,\n",
    "        '2 - High': 2,\n",
    "        '4 - Low': 4,\n",
    "        '1 - Critical': 1\n",
    "    }\n",
    "    \n",
    "}\n",
    "tickets = tickets.replace(cleanup_nums)\n",
    "tickets.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "7730453c",
   "metadata": {
    "_cell_guid": "2f9001a4-6efd-46e5-8445-32aeb86be6cd",
    "_uuid": "94d1290d-d153-4899-a705-ade608c1e0c9",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:51.258571Z",
     "iopub.status.busy": "2023-07-04T15:07:51.258138Z",
     "iopub.status.idle": "2023-07-04T15:07:51.284488Z",
     "shell.execute_reply": "2023-07-04T15:07:51.283123Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.04977,
     "end_time": "2023-07-04T15:07:51.287109",
     "exception": false,
     "start_time": "2023-07-04T15:07:51.237339",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>avg_priority</th>\n",
       "      <th>total_ticket</th>\n",
       "      <th>avg_interactions</th>\n",
       "      <th>avg_days_taken</th>\n",
       "      <th>avg_reopens</th>\n",
       "      <th>avg_impact</th>\n",
       "      <th>avg_state_Closed</th>\n",
       "      <th>avg_state_Open</th>\n",
       "      <th>made_sla_True</th>\n",
       "      <th>avg_open_True</th>\n",
       "      <th>avg_first_res_False</th>\n",
       "      <th>avg_first_res_True</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>requester_id</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2.0</th>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4.0</th>\n",
       "      <td>3</td>\n",
       "      <td>17</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>44.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5.0</th>\n",
       "      <td>3</td>\n",
       "      <td>13</td>\n",
       "      <td>3.384615</td>\n",
       "      <td>5.384615</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6.0</th>\n",
       "      <td>3</td>\n",
       "      <td>16</td>\n",
       "      <td>5.125000</td>\n",
       "      <td>114.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7.0</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              avg_priority  total_ticket  avg_interactions  avg_days_taken  \\\n",
       "requester_id                                                                 \n",
       "2.0                      4             3          1.000000        1.000000   \n",
       "4.0                      3            17          9.000000       44.000000   \n",
       "5.0                      3            13          3.384615        5.384615   \n",
       "6.0                      3            16          5.125000      114.000000   \n",
       "7.0                      3             3          1.000000        0.000000   \n",
       "\n",
       "              avg_reopens  avg_impact  avg_state_Closed  avg_state_Open  \\\n",
       "requester_id                                                              \n",
       "2.0                   0.0           2                 1               0   \n",
       "4.0                   0.0           1                 0               1   \n",
       "5.0                   0.0           1                 0               1   \n",
       "6.0                   0.0           1                 0               1   \n",
       "7.0                   0.0           1                 1               0   \n",
       "\n",
       "              made_sla_True  avg_open_True  avg_first_res_False  \\\n",
       "requester_id                                                      \n",
       "2.0                       1              1                    1   \n",
       "4.0                       1              1                    1   \n",
       "5.0                       1              1                    1   \n",
       "6.0                       1              1                    1   \n",
       "7.0                       1              1                    1   \n",
       "\n",
       "              avg_first_res_True  \n",
       "requester_id                      \n",
       "2.0                            0  \n",
       "4.0                            0  \n",
       "5.0                            0  \n",
       "6.0                            0  \n",
       "7.0                            0  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# encode categorical features with less number of unique values using one hot encoding\n",
    "tickets = pd.get_dummies(tickets, columns=[\"avg_state\", \"made_sla\", \"avg_open\", \"avg_first_res\"])\n",
    "tickets.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "f543ea31",
   "metadata": {
    "_cell_guid": "4fcdc325-44aa-4704-8aba-20bb8090ea08",
    "_uuid": "5c1d2964-1be8-45a9-8df2-7093f19bf89f",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:51.328793Z",
     "iopub.status.busy": "2023-07-04T15:07:51.328382Z",
     "iopub.status.idle": "2023-07-04T15:07:51.337100Z",
     "shell.execute_reply": "2023-07-04T15:07:51.335871Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.032839,
     "end_time": "2023-07-04T15:07:51.339821",
     "exception": false,
     "start_time": "2023-07-04T15:07:51.306982",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# split features and the target variable\n",
    "class_names = ['Bad', 'Average', 'Good']\n",
    "feature_cols = filter(lambda x: x != 'avg_impact', tickets.columns)\n",
    "X = tickets[feature_cols]\n",
    "y = tickets.avg_impact"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "ce80af27",
   "metadata": {
    "_cell_guid": "1a5be67e-37f8-49e5-a393-ab64a88feb2c",
    "_uuid": "b47b3fac-4ea0-493a-9a77-ba3d84d19aac",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:51.380481Z",
     "iopub.status.busy": "2023-07-04T15:07:51.380078Z",
     "iopub.status.idle": "2023-07-04T15:07:51.649886Z",
     "shell.execute_reply": "2023-07-04T15:07:51.648704Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.295002,
     "end_time": "2023-07-04T15:07:51.654158",
     "exception": false,
     "start_time": "2023-07-04T15:07:51.359156",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 480x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# explore the correlation between features\n",
    "plt.matshow(X.corr())\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1d6a4562",
   "metadata": {
    "_cell_guid": "6604b807-cc8e-46da-93e7-e50baa60a66c",
    "_uuid": "a90bd193-ac10-4247-bbac-f933f86b0648",
    "papermill": {
     "duration": 0.022293,
     "end_time": "2023-07-04T15:07:51.719733",
     "exception": false,
     "start_time": "2023-07-04T15:07:51.697440",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "There is no highly correlated features in the dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7588105a",
   "metadata": {
    "_cell_guid": "501fb047-1a97-4670-acbc-bd0a7a2d8dc0",
    "_uuid": "05836a7e-3831-4727-83a6-3e170b64166f",
    "papermill": {
     "duration": 0.020582,
     "end_time": "2023-07-04T15:07:51.759966",
     "exception": false,
     "start_time": "2023-07-04T15:07:51.739384",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Splitting train, test datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "6472a377",
   "metadata": {
    "_cell_guid": "c967e10f-7e41-42fa-8cd0-994c4c3e033d",
    "_uuid": "36a7b5dc-dca7-4f8e-ab0c-481c93e905a0",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:51.802130Z",
     "iopub.status.busy": "2023-07-04T15:07:51.801639Z",
     "iopub.status.idle": "2023-07-04T15:07:52.117354Z",
     "shell.execute_reply": "2023-07-04T15:07:52.115828Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.339815,
     "end_time": "2023-07-04T15:07:52.120399",
     "exception": false,
     "start_time": "2023-07-04T15:07:51.780584",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X_train size: 1017\n",
      "X_test size: 4072\n",
      "y_train Good:  0\n",
      "y_train Average:  14\n",
      "y_train Bad:  990\n"
     ]
    }
   ],
   "source": [
    "# splitting using stratified split\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=1, stratify=y)\n",
    "print('X_train size:', len(X_train))\n",
    "print('X_test size:', len(X_test))\n",
    "print('y_train Good: ', len(y_train.loc[y_train == 3]))\n",
    "print('y_train Average: ', len(y_train.loc[y_train == 2]))\n",
    "print('y_train Bad: ', len(y_train.loc[y_train == 1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "c2763d13",
   "metadata": {
    "_cell_guid": "0ac02286-cb39-42db-bdad-4c2b97bfd6e1",
    "_uuid": "340e6b80-2cb8-4f64-b14a-f276806ad92f",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:52.163934Z",
     "iopub.status.busy": "2023-07-04T15:07:52.163531Z",
     "iopub.status.idle": "2023-07-04T15:07:52.894602Z",
     "shell.execute_reply": "2023-07-04T15:07:52.893591Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.75576,
     "end_time": "2023-07-04T15:07:52.897909",
     "exception": false,
     "start_time": "2023-07-04T15:07:52.142149",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X_train size: 2970\n",
      "X_test size: 4072\n",
      "y_train Good:  990\n",
      "y_train Average:  990\n",
      "y_train Bad:  990\n"
     ]
    }
   ],
   "source": [
    "# using SMOTE up-sample the minority classes\n",
    "import imblearn\n",
    "from imblearn.over_sampling import SMOTE\n",
    "\n",
    "oversample = SMOTE()\n",
    "X_train, y_train = oversample.fit_resample(X_train, y_train)\n",
    "print('X_train size:', len(X_train))\n",
    "print('X_test size:', len(X_test))\n",
    "print('y_train Good: ', len(y_train.loc[y_train == 2]))\n",
    "print('y_train Average: ', len(y_train.loc[y_train == 1]))\n",
    "print('y_train Bad: ', len(y_train.loc[y_train == 0]))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7fddd56a",
   "metadata": {
    "_cell_guid": "5344e98c-c5e5-4446-86e9-5c1186bfd261",
    "_uuid": "5512a846-0759-4c92-aaed-533b159040c0",
    "papermill": {
     "duration": 0.020171,
     "end_time": "2023-07-04T15:07:52.940555",
     "exception": false,
     "start_time": "2023-07-04T15:07:52.920384",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Training Models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "4f5f6e9e",
   "metadata": {
    "_cell_guid": "7beffa1e-a5e4-4f3d-a3fb-542c1f4f9253",
    "_uuid": "48d4ae4b-6bc4-4f0b-b51d-38566a2c37eb",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:52.985583Z",
     "iopub.status.busy": "2023-07-04T15:07:52.985166Z",
     "iopub.status.idle": "2023-07-04T15:07:52.990469Z",
     "shell.execute_reply": "2023-07-04T15:07:52.989172Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.032466,
     "end_time": "2023-07-04T15:07:52.993345",
     "exception": false,
     "start_time": "2023-07-04T15:07:52.960879",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# importing libraries needed to evaluate performance\n",
    "from sklearn import metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "bb97a4fb",
   "metadata": {
    "_cell_guid": "1c27b7e3-9ad7-4deb-94cd-67cddc5d741c",
    "_uuid": "a06230c5-ad0b-477b-8d8b-27e8145e6de5",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:53.036752Z",
     "iopub.status.busy": "2023-07-04T15:07:53.035438Z",
     "iopub.status.idle": "2023-07-04T15:07:53.041541Z",
     "shell.execute_reply": "2023-07-04T15:07:53.040400Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.030589,
     "end_time": "2023-07-04T15:07:53.044401",
     "exception": false,
     "start_time": "2023-07-04T15:07:53.013812",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def plotConfusionMatrix(actual, pred):\n",
    "    cm = metrics.confusion_matrix(actual, pred)\n",
    "    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)\n",
    "    disp.plot()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1d495064",
   "metadata": {
    "_cell_guid": "4f944480-84f0-4cf5-8401-1d51ac91ca81",
    "_uuid": "77bcf3da-d843-4230-a27e-d40e7a55038c",
    "papermill": {
     "duration": 0.019593,
     "end_time": "2023-07-04T15:07:53.083928",
     "exception": false,
     "start_time": "2023-07-04T15:07:53.064335",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Using Decision Trees"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "9764ed62",
   "metadata": {
    "_cell_guid": "16bf2797-b4a9-4706-a32f-3bd1e95f3ab0",
    "_uuid": "ff853b6a-504e-4616-8b95-5d5bf510b7a7",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:53.126000Z",
     "iopub.status.busy": "2023-07-04T15:07:53.125352Z",
     "iopub.status.idle": "2023-07-04T15:07:53.475576Z",
     "shell.execute_reply": "2023-07-04T15:07:53.473888Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.374496,
     "end_time": "2023-07-04T15:07:53.478402",
     "exception": false,
     "start_time": "2023-07-04T15:07:53.103906",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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4kS4uLjAysoKCQkJaNmyJSwtLaFSqbBw4UJMnz4d9vb2GDp0KMrKypCSkoL8/HzMnDkTY8eOxfz58zFx4kS89957uHz5Mv7xj3/U63rbtWuHyspKrF69GsOHD8eBAwfw2Wef1Whnbm6OadOm4ZNPPoG5uTneeust9OjRQ0pw3n//fYSFhcHT0xMvvvgiTExMcOLECZw8eRIfffRR/T8IIjIIrnIiMhIKhQI7d+7EM888g7/85S9o3749XnrpJVy+fFlalRQeHo73338f8+bNQ2BgIK5cuYI333xTZ78LFizArFmz8P7778PPzw/h4eHIzc0FUD0/5ZNPPsG6devg4eGBkSNHAgBee+01fP7554iJiUFAQAD69u2LmJgYaZm3ra0tvv/+e5w5cwbdunXD/PnzsWTJknpdb9euXbF8+XIsWbIEnTp1QmxsLKKjo2u0s7a2xrx58zB27FiEhITAysoKcXFx0vEhQ4bghx9+QGJiIrp3744ePXpg+fLl8PLyqlc8RGRYCtEQg9lEREREBsQKDREREckeExoiIiKSPSY0REREJHtMaIiIiEj2mNAQERGR7DGhISIiItljQkNERESyx4SGiIiIZI8JDREREckeExoiIiKSPSY0REREJHv/D27csH1ebPz5AAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         Bad       0.72      0.66      0.69        50\n",
      "     Average       1.00      1.00      1.00      3965\n",
      "        Good       0.95      1.00      0.97        57\n",
      "\n",
      "    accuracy                           0.99      4072\n",
      "   macro avg       0.89      0.89      0.89      4072\n",
      "weighted avg       0.99      0.99      0.99      4072\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier\n",
    "dt_model = DecisionTreeClassifier(max_depth = 7)\n",
    "dt_model = dt_model.fit(X_train, y_train)\n",
    "y_pred = dt_model.predict(X_test)\n",
    "\n",
    "plotConfusionMatrix(y_test, y_pred);\n",
    "print(metrics.classification_report(y_test, y_pred, target_names=class_names))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e9f2d534",
   "metadata": {
    "_cell_guid": "aefb63c6-7c53-49de-a333-72553ae92e94",
    "_uuid": "272feff4-f23b-4fc4-bd9b-4a6fcbd34fae",
    "papermill": {
     "duration": 0.020787,
     "end_time": "2023-07-04T15:07:53.520770",
     "exception": false,
     "start_time": "2023-07-04T15:07:53.499983",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Using Naive Bayes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "b5128eab",
   "metadata": {
    "_cell_guid": "fcae445e-00d8-4399-8652-a58ecb934622",
    "_uuid": "8f03e0bb-dc6e-4913-9ad1-aa4e2c206421",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:53.563664Z",
     "iopub.status.busy": "2023-07-04T15:07:53.563192Z",
     "iopub.status.idle": "2023-07-04T15:07:53.915022Z",
     "shell.execute_reply": "2023-07-04T15:07:53.913552Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.377065,
     "end_time": "2023-07-04T15:07:53.918304",
     "exception": false,
     "start_time": "2023-07-04T15:07:53.541239",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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4kS4uLjAysoKCQkJaNmyJSwtLaFSqbBw4UJMnz4d9vb2GDp0KMrKypCSkoL8/HzMnDkTY8eOxfz58zFx4kS89957uHz5Mv7xj3/U63rbtWuHyspKrF69GsOHD8eBAwfw2Wef1Whnbm6OadOm4ZNPPoG5uTneeust9OjRQ0pw3n//fYSFhcHT0xMvvvgiTExMcOLECZw8eRIfffRR/T8IIjIIrnIiMhIKhQI7d+7EM888g7/85S9o3749XnrpJVy+fFlalRQeHo73338f8+bNQ2BgIK5cuYI333xTZ78LFizArFmz8P7778PPzw/h4eHIzc0FUD0/5ZNPPsG6devg4eGBkSNHAgBee+01fP7554iJiUFAQAD69u2LmJgYaZm3ra0tvv/+e5w5cwbdunXD/PnzsWTJknpdb9euXbF8+XIsWbIEnTp1QmxsLKKjo2u0s7a2xrx58zB27FiEhITAysoKcXFx0vEhQ4bghx9+QGJiIrp3744ePXpg+fLl8PLyqlc8RGRYCtEQg9lEREREBsQKDREREckeExoiIiKSPSY0REREJHtMaIiIiEj2mNAQERGR7DGhISIiItljQkNERESyx4SGiIiIZI8JDREREckeExoiIiKSPSY0REREJHv/D27csH1ebPz5AAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         Bad       0.72      0.66      0.69        50\n",
      "     Average       1.00      1.00      1.00      3965\n",
      "        Good       0.95      1.00      0.97        57\n",
      "\n",
      "    accuracy                           0.99      4072\n",
      "   macro avg       0.89      0.89      0.89      4072\n",
      "weighted avg       0.99      0.99      0.99      4072\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.naive_bayes import GaussianNB\n",
    "gnb = GaussianNB().fit(X_train, y_train)\n",
    "gnb_predictions = gnb.predict(X_test)\n",
    "  \n",
    "plotConfusionMatrix(y_test, y_pred);\n",
    "print(metrics.classification_report(y_test, y_pred, target_names=class_names))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d99e0fe2",
   "metadata": {
    "_cell_guid": "e27adb77-ebb5-49f3-8a17-1a3f716780b4",
    "_uuid": "24eae3ac-df98-4148-a4ed-842535a8d851",
    "papermill": {
     "duration": 0.021629,
     "end_time": "2023-07-04T15:07:53.961109",
     "exception": false,
     "start_time": "2023-07-04T15:07:53.939480",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Using Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "7ac4a230",
   "metadata": {
    "_cell_guid": "2d6d1684-57a1-49e2-a078-9e6aeb8fcfc1",
    "_uuid": "b8974b4d-3450-490f-8c7d-82ed1c2974cc",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:54.008601Z",
     "iopub.status.busy": "2023-07-04T15:07:54.008099Z",
     "iopub.status.idle": "2023-07-04T15:07:54.599753Z",
     "shell.execute_reply": "2023-07-04T15:07:54.597845Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.618032,
     "end_time": "2023-07-04T15:07:54.602806",
     "exception": false,
     "start_time": "2023-07-04T15:07:53.984774",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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F154ARcuXEBxcTGuX7+OiIgIjfvbEBERNSSdHnug44JiOaj1ouCUlBR069atyn6lUonCwqrPdiEiIqI6wDsFa1XrDk2bNm2QmJhYZf8PP/wAb2/vusiJiIiI7sc1NFrVukPz2muvYebMmSguLoYQAseOHcOXX34pPQCSiIiIqKHVuqCZPHkyysvLsXDhQty5cwfjx49HixYtsGbNGjz33HP1kSMREVGTxxvrafdIN9abNm0apk2bhlu3bkGtVsPBwaGu8yIiIqJ78dEHWul0p2B7e/u6yoOIiIjokdW6oGnTpo3WJ1FfvvzwW5wTERFRLel66TU7NJpCQkI0XpeVleHkyZOIiYnBa6+9Vld5ERER0b045aRVrQuauXPnVrv/P//5D06cOKFzQkRERES19UjPcqrOsGHD8PXXX9fVcERERHQv3odGK50WBd/rq6++gq2tbV0NR0RERPfgZdva1bqg6datm8aiYCEE0tPTkZWVhY8//rhOkyMiIiKqiVoXNKNHj9Z4bWBggObNm6Nfv37o0KFDXeVFREREVGO1KmjKy8vRunVrDB06FE5OTvWVExEREd2PVzlpVatFwUZGRnjllVdQUlJSX/kQERFRNe6uodFla8xqfZWTv78/Tp48WR+5EBERET2SWq+hmTFjBubPn4/r16/D19cXFhYWGsc7d+5cZ8kRERHRPRp5l0UXNS5oXnzxRaxevRrjxo0DAMyZM0c6plAoIISAQqFARUVF3WdJRETU1HENjVY1Lmi2bt2K9957DykpKfWZDxEREVGt1bigEaKytHNzc6u3ZIiIiKh6vLGedrVaQ6PtKdtERERUjzjlpFWtCpr27ds/tKj566+/dEqIiIiIqLZqVdC88847UKlU9ZULERERPQCnnLSrVUHz3HPPwcHBob5yISIiogfRw5TToUOH8OGHHyIhIQFpaWmIjo7WeATSpEmTsHXrVo1z/P39ER8fL70uKSnBggUL8OWXX6KoqAgDBw7Exx9/jJYtW0ox2dnZmDNnDr777jsAwMiRI7Fu3To0a9asxrnW+MZ6XD9DRETUtBQWFqJLly5Yv379A2MCAwORlpYmbXv37tU4HhISgujoaERFRSEuLg4FBQUICgrSuM3L+PHjkZiYiJiYGMTExCAxMRHBwcG1yrXWVzkRERGRHtRRhyYvL09jt1KphFKprPaUYcOG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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         Bad       0.69      0.96      0.80        50\n",
      "     Average       0.99      0.94      0.97      3965\n",
      "        Good       0.15      0.65      0.24        57\n",
      "\n",
      "    accuracy                           0.94      4072\n",
      "   macro avg       0.61      0.85      0.67      4072\n",
      "weighted avg       0.98      0.94      0.95      4072\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "logreg = LogisticRegression(random_state=16)\n",
    "logreg.fit(X_train, y_train)\n",
    "y_pred = logreg.predict(X_test)\n",
    "plotConfusionMatrix(y_test, y_pred);\n",
    "print(metrics.classification_report(y_test, y_pred, target_names=class_names))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "79ebc8f6",
   "metadata": {
    "_cell_guid": "9365e9e3-4d49-4bd1-8b07-93a4c999543f",
    "_uuid": "a6c3fc34-2345-47b6-bdb1-f6062418b713",
    "papermill": {
     "duration": 0.021205,
     "end_time": "2023-07-04T15:07:54.645621",
     "exception": false,
     "start_time": "2023-07-04T15:07:54.624416",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Using Random Forests"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "03a3213e",
   "metadata": {
    "_cell_guid": "1b3dd747-7189-4015-9446-cca0914135ed",
    "_uuid": "304ac56d-c452-4910-aedd-722a2e96b358",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:54.691528Z",
     "iopub.status.busy": "2023-07-04T15:07:54.691120Z",
     "iopub.status.idle": "2023-07-04T15:07:55.469602Z",
     "shell.execute_reply": "2023-07-04T15:07:55.465140Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.80562,
     "end_time": "2023-07-04T15:07:55.472774",
     "exception": false,
     "start_time": "2023-07-04T15:07:54.667154",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         Bad       0.78      0.72      0.75        50\n",
      "     Average       1.00      1.00      1.00      3965\n",
      "        Good       0.95      1.00      0.97        57\n",
      "\n",
      "    accuracy                           0.99      4072\n",
      "   macro avg       0.91      0.91      0.91      4072\n",
      "weighted avg       0.99      0.99      0.99      4072\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "rf = RandomForestClassifier()\n",
    "rf.fit(X_train, y_train)\n",
    "y_pred = rf.predict(X_test)\n",
    "plotConfusionMatrix(y_test, y_pred);\n",
    "print(metrics.classification_report(y_test, y_pred, target_names=class_names))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5bb6f46b",
   "metadata": {
    "_cell_guid": "264003a6-1cd4-42fd-ac51-79fe51b114bf",
    "_uuid": "6e095649-9711-496e-a320-b9b4505524e5",
    "papermill": {
     "duration": 0.025251,
     "end_time": "2023-07-04T15:07:55.524830",
     "exception": false,
     "start_time": "2023-07-04T15:07:55.499579",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Using Support Vector Machines"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "bd569c3e",
   "metadata": {
    "_cell_guid": "86c28a04-f393-4a40-a35d-5dbb6abfe073",
    "_uuid": "70ecc378-9cff-4798-a944-66ea23d0a8a3",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:55.571366Z",
     "iopub.status.busy": "2023-07-04T15:07:55.570796Z",
     "iopub.status.idle": "2023-07-04T15:07:56.860507Z",
     "shell.execute_reply": "2023-07-04T15:07:56.858873Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 1.316483,
     "end_time": "2023-07-04T15:07:56.863417",
     "exception": false,
     "start_time": "2023-07-04T15:07:55.546934",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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q6sr+vTpgy1btsDGxkZqv3TpUpiYmGD48OEoKipC3759ER0dDWNjY6lNTEwMpkyZIq2GGjx4MFauXFmreJnQEBERycCTDBs9en7t2muv6VhYWGDXrl1/2Y+5uTlWrFiBFStWPLaNvb09Nm3aVKv4HsWEhoiISA6e9piTzDChISIikgMdKzTQ5VwZ4ConIiIikj1WaIiIiGTg4bv9Pun5howJDRERkQw87UnBcsMhJyIiIpI9VmiIiIjkQCh0m9hr4BUaJjREREQywDk02nHIiYiIiGSPFRoiIiI54I31tGJCQ0REJANc5aRdjRKazz77rMYdTpky5YmDISIiInoSNUpoli5dWqPOFAoFExoiIqL6YuDDRrqoUUKTlpZW33EQERGRFhxy0u6JVzmVlpbi/PnzKC8vr8t4iIiIqDqiDjYDVuuE5v79+xg7diwsLS3Rtm1bXLt2DUDl3JmFCxfWeYBEREREf6XWCc2HH36I3377Dfv27YO5ubm0v1+/ftiyZUudBkdEREQPKOpgM1y1Xra9bds2bNmyBV27doVC8ecXx9fXF5cuXarT4IiIiOh/eB8arWpdocnOzoaTk1OV/YWFhRoJDhEREdHTUuuEpkuXLtixY4f0+kESs3btWgQGBtZdZERERPQnTgrWqtZDTlFRUQgODsa5c+dQXl6O5cuX4+zZszh8+DD2799fHzESERERn7atVa0rNN26dcPBgwdx//59tGrVCrt374azszMOHz4MPz+/+oiRiIiISKsnepZT+/btsWHDhrqOhYiIiB5DiMpNl/MN2RMlNBUVFYiLi0NqaioUCgV8fHwwZMgQmJjwWZdERET1gquctKp1BnLmzBkMGTIEmZmZ8Pb2BgD88ccfaNy4MX744Qe0b9++zoMkIiIi0qbWc2jefvtttG3bFtevX8eJEydw4sQJpKeno0OHDhg/fnx9xEhEREQPJgXrshmwWldofvvtNxw/fhx2dnbSPjs7OyxYsABdunSp0+CIiIiokkJUbrqcb8hqXaHx9vbGrVu3quzPyspC69at6yQoIiIiegTvQ6NVjRKavLw8aYuMjMSUKVPwzTff4Pr167h+/Tq++eYbhIeH45NPPqnveImIiIiqqNGQU6NGjTQeayCEwPDhw6V94n9rwQYNGoSKiop6CJOIiKiB4431tKpRQrN37976joOIiIi04bJt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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         Bad       0.65      0.98      0.78        50\n",
      "     Average       1.00      1.00      1.00      3965\n",
      "        Good       0.94      0.84      0.89        57\n",
      "\n",
      "    accuracy                           0.99      4072\n",
      "   macro avg       0.86      0.94      0.89      4072\n",
      "weighted avg       0.99      0.99      0.99      4072\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)\n",
    "y_pred = svm_model_linear.predict(X_test)\n",
    "plotConfusionMatrix(y_test, y_pred);\n",
    "print(metrics.classification_report(y_test, y_pred, target_names=class_names))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e8a2c518",
   "metadata": {
    "_cell_guid": "a08a3997-4f84-4211-a2a4-7561a948a279",
    "_uuid": "6707cca3-71f9-4c32-99b7-7ad22f1d628b",
    "papermill": {
     "duration": 0.022337,
     "end_time": "2023-07-04T15:07:56.908937",
     "exception": false,
     "start_time": "2023-07-04T15:07:56.886600",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Using K-Nearest Neighbors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "f5737051",
   "metadata": {
    "_cell_guid": "4dd4e1a6-b47d-4319-bc1a-0a9695d7b4fc",
    "_uuid": "ae4286ed-3ffd-4ef5-bab6-1301c3babc5b",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:56.956658Z",
     "iopub.status.busy": "2023-07-04T15:07:56.956225Z",
     "iopub.status.idle": "2023-07-04T15:07:57.435677Z",
     "shell.execute_reply": "2023-07-04T15:07:57.433879Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.506259,
     "end_time": "2023-07-04T15:07:57.438335",
     "exception": false,
     "start_time": "2023-07-04T15:07:56.932076",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         Bad       0.03      0.40      0.05        50\n",
      "     Average       0.98      0.77      0.86      3965\n",
      "        Good       0.11      0.35      0.16        57\n",
      "\n",
      "    accuracy                           0.76      4072\n",
      "   macro avg       0.37      0.51      0.36      4072\n",
      "weighted avg       0.96      0.76      0.84      4072\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "knn = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train)\n",
    "y_pred = knn.predict(X_test) \n",
    "plotConfusionMatrix(y_test, y_pred);\n",
    "print(metrics.classification_report(y_test, y_pred, target_names=class_names))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "76937b5b",
   "metadata": {
    "_cell_guid": "0dac5b23-5e14-4b5c-a6db-814e15b6c2ed",
    "_uuid": "706eefd9-84a0-43b9-8af4-50f8fefba1a3",
    "papermill": {
     "duration": 0.024265,
     "end_time": "2023-07-04T15:07:57.486380",
     "exception": false,
     "start_time": "2023-07-04T15:07:57.462115",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### AdaBoost with SVM base learners"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "2595e456",
   "metadata": {
    "_cell_guid": "645ead4a-9ef7-42ff-aceb-1f3e3eb231cb",
    "_uuid": "edb84004-7818-4d41-beb0-e305d3591f6c",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:07:57.535719Z",
     "iopub.status.busy": "2023-07-04T15:07:57.535305Z",
     "iopub.status.idle": "2023-07-04T15:09:38.149148Z",
     "shell.execute_reply": "2023-07-04T15:09:38.147694Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 100.654153,
     "end_time": "2023-07-04T15:09:38.164252",
     "exception": false,
     "start_time": "2023-07-04T15:07:57.510099",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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HXD79u0l2qnVapw6dQovv/wyAODy5ct4+PAhmjRpAqD463b58uVyfa2J6MXDhIaoiujevTsCAwMxYMAALF26FN7e3rhz5w727NmDAQMGICAgANOmTcPIkSMREBCADh06ICIiAhcuXECDBg1K7dPS0hJz587FnDlzYGFhgfbt2yMtLQ0XLlzAmDFj4OzsDCsrK0RFRaFOnTqwtLSESqVCaGgopk6dCnt7e/Tu3RsFBQU4deoU0tPTMWPGDAwZMgTz5s3DmDFj8N577+HGjRv45JNPynW9DRs2hFqtxqpVq9C3b18cOXIEn3/+eYl25ubmmDJlCj777DOYm5tj8uTJaNu2rZTgvP/++wgODoaHhwdef/11mJiY4OzZszh37hwWLVpU/m8EERkFVzkRVREKhQJ79uxBp06d8MYbb6Bx48YYPHgwbty4Ia1KCgkJwfvvv4+5c+fC398fN2/exFtvvaW33/nz52PmzJl4//334ePjg5CQEKSmpgIonp/y2WefYd26dXB3d0f//v0BAGPHjsWXX36J8PBw+Pn5oXPnzggPD5eWedva2uKnn37CxYsX0apVK8ybNw9Lly4t1/W2bNkSy5cvx9KlS9GsWTNEREQgLCysRDtra2vMnTsXQ4YMQWBgIKysrLB161bpeK9evbBr1y5ER0ejdevWaNu2LZYvXw5PT89yxUNExqUQFTGYTURERGRErNAQERGR7DGhISIiItljQkNERESyx4SGiIiIZI8JDREREckeExoiIiKSPSY0REREJHtMaIiIiEj2mNAQERGR7DGhISIiItljQkNERESy9/8aOLAabUcn/wAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         Bad       0.17      0.98      0.29        50\n",
      "     Average       0.99      0.67      0.80      3965\n",
      "        Good       0.02      0.44      0.04        57\n",
      "\n",
      "    accuracy                           0.67      4072\n",
      "   macro avg       0.39      0.70      0.38      4072\n",
      "weighted avg       0.96      0.67      0.78      4072\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import AdaBoostClassifier\n",
    "\n",
    "svc=SVC(probability=True, kernel='linear')\n",
    "abc =AdaBoostClassifier(n_estimators=50, estimator=svc,learning_rate=1)\n",
    "model = abc.fit(X_train, y_train)\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "plotConfusionMatrix(y_test, y_pred);\n",
    "print(metrics.classification_report(y_test, y_pred, target_names=class_names))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8be976d5",
   "metadata": {
    "_cell_guid": "a39c88bd-3291-4f74-ae77-6c8a1f3e3cfb",
    "_uuid": "6e2b06b1-3e0d-4721-a2a3-ab7eab49b5fe",
    "papermill": {
     "duration": 0.024283,
     "end_time": "2023-07-04T15:09:38.212153",
     "exception": false,
     "start_time": "2023-07-04T15:09:38.187870",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Using XGBoost Decision Trees"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "332fcb36",
   "metadata": {
    "_cell_guid": "514c59ae-f7c8-4651-8a94-e80f658012b4",
    "_uuid": "d2398aa8-bb55-4fee-8c16-eedaf9463778",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2023-07-04T15:09:38.263392Z",
     "iopub.status.busy": "2023-07-04T15:09:38.262922Z",
     "iopub.status.idle": "2023-07-04T15:09:39.132624Z",
     "shell.execute_reply": "2023-07-04T15:09:39.129990Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.89953,
     "end_time": "2023-07-04T15:09:39.135642",
     "exception": false,
     "start_time": "2023-07-04T15:09:38.236112",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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p7Nnz0IIAR8fH639zs7OKCkpAQBMnjwZQUFBOH78ONLT06WkYsOGDejQoQOOHDmCbt264eOPP8bYsWMxadIkAMDMmTORlJSEjz/+GH379sWuXbuQmpqKCxcuoHnz5gCAyMhIDB48+L7xvf3225g5c6b0Oj8/v9EkNfk3TVFZATg01f5XnMq5Ark5RvXHkMDPuzEpLTbFhTQreLQs0drf4akCeHqXInKys4EiMz589IFuRpk2KxQKrdeHDx9GSkoKOnTogNLSUqSmpsLT01MrmWjfvj2aNGmC1NRUAEBqaip69uyp1U/Pnj21jrdo0UJKZgBArVbrjEupVMLe3l5raywqyk1w5rg1nnymQGv/k88U4PTRmqtpJF/8vBsPcwsNPL1LcDPbXGt/cOgN/HncGump1gaKjBobo/qnkre3NxQKBf744w+t/a1atQIAWFlZAQCEENWSnpr239vm7uOihky3pj7pL99/7oy3Ps3An8etkHrUBs++dAMuzcqx7SsnQ4dGDYCft3EaP+8yDu1SIfuqBZo4VeD/pmXC2rYSu77963O1tq3E08/l4vP3m+voieqMk4J1MqqExsnJCQMGDMCKFSswderU+86jad++PS5duoSMjAypSnP69Gnk5eXB19cXAODr64vExES8/PLL0nkHDhyQjt/p4+rVq/Dw8AAAHDx4sCEvT/b2/tcBdg6VGP3mNTi6VOBimiX+8VJLZPOeJEaJn7dxcnYvw99XpMPeoQJ5N83wxzEbvDm8HbKv/LXKqffQm4BCYM8PjgaM1AgJAPosvTbufMa4EhoAWLlyJXr27ImAgABERESgU6dOMDExwZEjR/DHH3/A398fQUFB6NSpE0aPHo1ly5ahoqICkyZNQu/evREQEAAAeOuttzBy5Eg8+eST6N+/P3788Ud8//332LVrFwAgKCgIPj4+ePnll7F48WLk5+dj3rx5hrx0Wdi63hlb13NMvbHg5218Fk1p9cA2P21qip82NX0E0TQunEOjm9HNoWndujV+++03BAUF4e2330bnzp0REBCA5cuXY/bs2fjXv/4FhUKBLVu2wMHBAc888wyCgoLQqlUrbN68Wepn+PDh+OSTT/DRRx+hQ4cOWL16NdatW4c+ffoAAExMTBAXF4fS0lI89dRTGD9+PBYuXGigqyYiImrcFKKmySD0SOTn50OlUqEPhsFM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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         Bad       0.75      0.86      0.80        50\n",
      "     Average       1.00      1.00      1.00      3965\n",
      "        Good       0.95      1.00      0.97        57\n",
      "\n",
      "    accuracy                           0.99      4072\n",
      "   macro avg       0.90      0.95      0.93      4072\n",
      "weighted avg       0.99      0.99      0.99      4072\n",
      "\n"
     ]
    }
   ],
   "source": [
    "import xgboost as xgb\n",
    "xgb_classifier = xgb.XGBClassifier()\n",
    "xgb_classifier.fit(X_train, y_train)\n",
    "y_pred = xgb_classifier.predict(X_test)\n",
    "\n",
    "plotConfusionMatrix(y_test, y_pred);\n",
    "print(metrics.classification_report(y_test, y_pred, target_names=class_names))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.10"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 134.508757,
   "end_time": "2023-07-04T15:09:40.385937",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-07-04T15:07:25.877180",
   "version": "2.4.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
