# Zindi-Uber-Traffic-Jam-Competition

This repo contains my major learnings from participating in the Zindi Uber Traffic Jam Competition. It is specifically adapted for new data scientists.

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### This notebook chronicles my major learnings from the Uber traffic prediction competition hosted on Zindi."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Initial Preprocessing "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this competition, the target value was not explicitly given in the training set so we needed to an initial preprocessing to obtain it. Special thanks to the community for providing us with the initial processing code. The code is replicated below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pylab as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "uber = pd.read_csv('train_revised.csv', low_memory=False)\n",
    "test = pd.read_csv('test_questions.csv', low_memory=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
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
       "      <th>ride_id</th>\n",
       "      <th>seat_number</th>\n",
       "      <th>payment_method</th>\n",
       "      <th>payment_receipt</th>\n",
       "      <th>travel_date</th>\n",
       "      <th>travel_time</th>\n",
       "      <th>travel_from</th>\n",
       "      <th>travel_to</th>\n",
       "      <th>car_type</th>\n",
       "      <th>max_capacity</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1442</td>\n",
       "      <td>15A</td>\n",
       "      <td>Mpesa</td>\n",
       "      <td>UZUEHCBUSO</td>\n",
       "      <td>17-10-17</td>\n",
       "      <td>7:15</td>\n",
       "      <td>Migori</td>\n",
       "      <td>Nairobi</td>\n",
       "      <td>Bus</td>\n",
       "      <td>49</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5437</td>\n",
       "      <td>14A</td>\n",
       "      <td>Mpesa</td>\n",
       "      <td>TIHLBUSGTE</td>\n",
       "      <td>19-11-17</td>\n",
       "      <td>7:12</td>\n",
       "      <td>Migori</td>\n",
       "      <td>Nairobi</td>\n",
       "      <td>Bus</td>\n",
       "      <td>49</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5710</td>\n",
       "      <td>8B</td>\n",
       "      <td>Mpesa</td>\n",
       "      <td>EQX8Q5G19O</td>\n",
       "      <td>26-11-17</td>\n",
       "      <td>7:05</td>\n",
       "      <td>Keroka</td>\n",
       "      <td>Nairobi</td>\n",
       "      <td>Bus</td>\n",
       "      <td>49</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>5777</td>\n",
       "      <td>19A</td>\n",
       "      <td>Mpesa</td>\n",
       "      <td>SGP18CL0ME</td>\n",
       "      <td>27-11-17</td>\n",
       "      <td>7:10</td>\n",
       "      <td>Homa Bay</td>\n",
       "      <td>Nairobi</td>\n",
       "      <td>Bus</td>\n",
       "      <td>49</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5778</td>\n",
       "      <td>11A</td>\n",
       "      <td>Mpesa</td>\n",
       "      <td>BM97HFRGL9</td>\n",
       "      <td>27-11-17</td>\n",
       "      <td>7:12</td>\n",
       "      <td>Migori</td>\n",
       "      <td>Nairobi</td>\n",
       "      <td>Bus</td>\n",
       "      <td>49</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   ride_id seat_number payment_method payment_receipt travel_date travel_time  \\\n",
       "0     1442         15A          Mpesa      UZUEHCBUSO    17-10-17        7:15   \n",
       "1     5437         14A          Mpesa      TIHLBUSGTE    19-11-17        7:12   \n",
       "2     5710          8B          Mpesa      EQX8Q5G19O    26-11-17        7:05   \n",
       "3     5777         19A          Mpesa      SGP18CL0ME    27-11-17        7:10   \n",
       "4     5778         11A          Mpesa      BM97HFRGL9    27-11-17        7:12   \n",
       "\n",
       "  travel_from travel_to car_type  max_capacity  \n",
       "0      Migori   Nairobi      Bus            49  \n",
       "1      Migori   Nairobi      Bus            49  \n",
       "2      Keroka   Nairobi      Bus            49  \n",
       "3    Homa Bay   Nairobi      Bus            49  \n",
       "4      Migori   Nairobi      Bus            49  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "uber.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "ride_id_dict = {} \n",
    "for ride_id in uber[\"ride_id\"]:\n",
    "    if not ride_id in ride_id_dict:\n",
    "        ride_id_dict[ride_id] = 1\n",
    "    else:\n",
    "        ride_id_dict[ride_id] += 1 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "uber = uber.drop(['seat_number', 'payment_method', 'payment_receipt', 'travel_to'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "uber.drop_duplicates(inplace=True)\n",
    "uber.reset_index(drop= True, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "uber[\"number_of_tickets\"]= np.zeros(len(uber))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(len(uber)):\n",
    "    ride_id = uber.loc[i][\"ride_id\"]\n",
    "    uber.at[i,\"number_of_tickets\"] = ride_id_dict[ride_id]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
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
       "      <th>ride_id</th>\n",
       "      <th>travel_date</th>\n",
       "      <th>travel_time</th>\n",
       "      <th>travel_from</th>\n",
       "      <th>car_type</th>\n",
       "      <th>max_capacity</th>\n",
       "      <th>number_of_tickets</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1442</td>\n",
       "      <td>17-10-17</td>\n",
       "      <td>7:15</td>\n",
       "      <td>Migori</td>\n",
       "      <td>Bus</td>\n",
       "      <td>49</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5437</td>\n",
       "      <td>19-11-17</td>\n",
       "      <td>7:12</td>\n",
       "      <td>Migori</td>\n",
       "      <td>Bus</td>\n",
       "      <td>49</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5710</td>\n",
       "      <td>26-11-17</td>\n",
       "      <td>7:05</td>\n",
       "      <td>Keroka</td>\n",
       "      <td>Bus</td>\n",
       "      <td>49</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>5777</td>\n",
       "      <td>27-11-17</td>\n",
       "      <td>7:10</td>\n",
       "      <td>Homa Bay</td>\n",
       "      <td>Bus</td>\n",
       "      <td>49</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5778</td>\n",
       "      <td>27-11-17</td>\n",
       "      <td>7:12</td>\n",
       "      <td>Migori</td>\n",
       "      <td>Bus</td>\n",
       "      <td>49</td>\n",
       "      <td>31.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   ride_id travel_date travel_time travel_from car_type  max_capacity  \\\n",
       "0     1442    17-10-17        7:15      Migori      Bus            49   \n",
       "1     5437    19-11-17        7:12      Migori      Bus            49   \n",
       "2     5710    26-11-17        7:05      Keroka      Bus            49   \n",
       "3     5777    27-11-17        7:10    Homa Bay      Bus            49   \n",
       "4     5778    27-11-17        7:12      Migori      Bus            49   \n",
       "\n",
       "   number_of_tickets  \n",
       "0                1.0  \n",
       "1                1.0  \n",
       "2                1.0  \n",
       "3                5.0  \n",
       "4               31.0  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "uber.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# EXPLORATORY DATA ANALYSIS"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this step, we aim to understand the provided data to see if there are patterns, structures, etc. that could be useful in our modelling phase. \n",
    "\n",
    "N.B: There is another EDA provided through the community. You can also check it out in the Uber Discussion."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 6249 entries, 0 to 6248\n",
      "Data columns (total 7 columns):\n",
      "ride_id              6249 non-null int64\n",
      "travel_date          6249 non-null object\n",
      "travel_time          6249 non-null object\n",
      "travel_from          6249 non-null object\n",
      "car_type             6249 non-null object\n",
      "max_capacity         6249 non-null int64\n",
      "number_of_tickets    6249 non-null float64\n",
      "dtypes: float64(1), int64(2), object(4)\n",
      "memory usage: 341.8+ KB\n"
     ]
    }
   ],
   "source": [
    "uber.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1111 entries, 0 to 1110\n",
      "Data columns (total 7 columns):\n",
      "ride_id         1111 non-null int64\n",
      "travel_date     1111 non-null object\n",
      "travel_time     1111 non-null object\n",
      "travel_from     1111 non-null object\n",
      "travel_to       1111 non-null object\n",
      "car_type        1111 non-null object\n",
      "max_capacity    1111 non-null int64\n",
      "dtypes: int64(2), object(5)\n",
      "memory usage: 60.8+ KB\n"
     ]
    }
   ],
   "source": [
    "test.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "From the above codes, we can see that there are no null values meaning this dataset is 'super-clean'. This is not the usual scenerio. Data-cleaning is usually an important step developing a good model.\n",
    "\n",
    "We can also see that our travel dates is treated as an object. We should reconsider changing it to datetype."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "uber['travel_date'] = pd.to_datetime(uber['travel_date'])\n",
    "test['travel_date'] = pd.to_datetime(test['travel_date'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "datetime64[ns]\n",
      "datetime64[ns]\n"
     ]
    }
   ],
   "source": [
    "print(uber['travel_date'].dtypes)\n",
    "print(test['travel_date'].dtypes)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let us start our exploration proper."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6c93e2e8>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6cee3b70>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(x='car_type',y='max_capacity',data=uber)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6cee3f98>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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18nB701kzqQiIlZIehhYCewErvMZVWbWVM19DUxjbuvesWNHBg4cSETQsWNH7rrrLr7whS80ax8tKZdnVb0ODKmn/haZ4x171rcDl+xjWdOAac3do5lZSzjssMMoLy8HMqfr3nTTTTz//PN57qrpfMsRM7MW9P7779O9e3cA/vCHP3DBBRfUfjZx4sTa6zimTJlC//79GTRoEDfccEM+Wt0n31bdzCzH/v73v1NSUsL27dtZt24dCxcubHD6TZs28dhjj1FRUYEkNm/e3EKdNo63OMzMcqxmV1VFRQVPPfUUY8eOpaH7BB5xxBG1Nzb83e9+x+GHH96C3e6fg8PMrAV9/vOfZ8OGDVRXV9OpUyd2795d+1nN8z06derEokWLuPjii3n88cc5//zz89VuvbyrysysBVVUVLBr1y569OjBZz/7WVauXMmOHTvYvn07CxYs4Itf/CIffPABH374ISNHjuT000/n+OOPz3fbdTg4zKxdaczps82t5hgHZJ4sOHv2bDp27Ejv3r0ZM2YMgwYNori4mCFDMieibt26lVGjRrF9+3YigjvvvLPFe26Ig8PMLMd27dr3pWe33nort9566171RYta73XOPsZhZmapODjMzCwVB4eZmaXi4DAzs1QcHGZmloqDw8zMUvHpuGbWrrx9y8BmXd4//GBZ6nmKioooKyujoKCgUdNv3ryZBx54gG9961sAVFZW8vLLL/P1r38dyNws8fbbb+fJJ59M3UtTeIvDzKyV27x5M3fffXfteGVlJQ888EDe+nFwmJnl0LZt2/jSl77E4MGDGTBgAA899BAAM2bM4KSTTmLgwIFUVFQAMHXqVG6//fbaeQcMGEBlZSVTpkzhzTffpKSkhMmTJzNlyhReeOEFSkpK9rqqfNu2bVx11VWccsopDBkyhCeeeILm5uAwM8uhp556imOPPZY///nPLF++vPaGhQUFBSxdupRvfvObdcKiPtOnT+e4446jvLyc2267jenTp3PGGWdQXl7O9ddfX2faadOmcc4557B48WKee+45Jk+ezLZt25r1Ozk4zMxyaODAgTz77LPceOONvPDCC3Tr1g2Aiy66CICTTz6ZysrKZlvfM888w/Tp0ykpKeGss85i+/btvP322822fPDBcTOznOrbty9Llixh3rx53HTTTQwbNgyAQw89FMg8j3znzp0A+7zN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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6c93eef0>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='max_capacity',data=uber,hue='car_type')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "From the data description, we already know that all buses and shuttles have the same maximum capacity. That was easily confirmed using the seaborn barplot.\n",
    "\n",
    "Another thing we explored was the numbers of buses and shuttles in the dataset using seaborn's countplot. We can see that they are rougly the same. \n",
    "\n",
    "We can confirmed if this holds true for the test set too"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6ca44400>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6c9e9940>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(x='car_type',y='max_capacity',data=test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6cab2278>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYgAAAELCAYAAADDZxFQAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAAGjhJREFUeJzt3Xu0VnW97/H3l4uibUQE8hhwWpTI0bhJy7LMbV6GiJm4FcnaJngZdFFH6JEtnkZtdnvbIbVjhkU51ISOFqZ5yc0xE9RMt5eFoqhQorF1HTnKRQgxNOB7/ngmuIApLGA961mw3q8xnvHM+Zu/OZ/vswbj+TB/8xaZiSRJm+tQ6wIkSW2TASFJKmVASJJKGRCSpFIGhCSplAEhSSplQEiSShkQkqRSBoQkqVSnWhewM3r27Jl1dXW1LkOSdilz5sxZmpm9ttVvlw6Iuro6Ghoaal2GJO1SIuI/m9PPISZJUikDQpJUyoCQJJXapY9BSBLA3/72NxobG1mzZk2tS2lTunTpQp8+fejcufMOrW9ASNrlNTY20rVrV+rq6oiIWpfTJmQmy5Yto7GxkX79+u3QNqo6xBQR+0bEbRGxICLmR8SnImK/iPhdRLxYvHcv+kZE/DAiFkbEsxExrJq1Sdp9rFmzhh49ehgOTUQEPXr02Km9qmofg7gGuDcz/xswBJgPTARmZWZ/YFYxDzAC6F+8xgFTq1ybpN2I4bClnf2bVC0gImIf4O+BGwAy893MXAGMBKYV3aYBpxTTI4HpWfEYsG9EHFCt+iRJW1fNPYiPAEuAn0XE0xFxfUR8ANg/MxcDFO8fLPr3Bl5tsn5j0baJiBgXEQ0R0bBkyZIqli9J22/u3LnMnDmz1mW0iGoepO4EDAMuzMzHI+Ia3htOKlO2L5RbNGReB1wHUF9fv8VyaXfx8QnTa11CmzHnyrNqXcIW1q5dS6dOW/6Ezp07l4aGBk488cQaVNWyqrkH0Qg0ZubjxfxtVALj9Q1DR8X7G036922yfh/gtSrWJ0kATJ8+ncGDBzNkyBC+/OUv85vf/IZPfvKTHHrooRx33HG8/vrrAEyaNIlx48Zx/PHHc9ZZW4bWu+++y7e//W1mzJjB0KFDmTFjBv3792fDaMf69es58MADWbp0KWPHjuWrX/0qRx55JAcddBD33HMPAOvWrWPChAkcdthhDB48mJ/+9Ket94fYTNX2IDLz/0XEqxExIDP/CBwLvFC8xgCTi/e7ilXuBi6IiF8CnwRWbhiKkqRqef7557n88st55JFH6NmzJ8uXLycieOyxx4gIrr/+eq644gq+//3vAzBnzhz+8Ic/sNdee22xrT322IPvfOc7NDQ0cO211wKwYMECbr75ZsaPH8/999/PkCFD6NmzJwCLFi3ioYce4qWXXuLoo49m4cKFTJ8+nW7duvHkk0/yzjvvcMQRR3D88cfv8KmqO6Pa10FcCNwcEXsALwNnU9lruTUizgVeAU4v+s4ETgQWAm8XfSWpqmbPns2oUaM2/mjvt99+zJs3jy984QssXryYd999d5Mf55NPPrk0HN7POeecw8iRIxk/fjw33ngjZ5/93k/b6NGj6dChA/379+cjH/kICxYs4L777uPZZ5/ltttuA2DlypW8+OKLu19AZOZcoL5k0bElfRM4v5r1SNLmMnOL00EvvPBCLr74Yk4++WQefPBBJk2atHHZBz7wge3aft++fdl///2ZPXs2jz/+ODfffPPGZZt/bkSQmUyZMoXhw4dv/5dpYd6LSVK7duyxx3LrrbeybNkyAJYvX87KlSvp3btyEuW0adO2tvoWunbtyqpVqzZpO++88zjzzDMZPXo0HTt23Nj+q1/9ivXr1/PSSy/x8ssvM2DAAIYPH87UqVP529/+BsCf/vQnVq9evTNfcYcZEJLatY997GN885vf5KijjmLIkCFcfPHFTJo0idNPP50jjzxy49BTcx199NG88MILGw9SQ2VY6q233tpkeAlgwIABHHXUUYwYMYKf/OQndOnShfPOO49DDjmEYcOGMXDgQL7yla+wdu3aFvu+2yMqIzu7pvr6+vSBQdpdeZrre7Z1muv8+fM5+OCDW6ma7dfQ0MBFF13Eww8/vLFt7NixnHTSSYwaNaqqn132t4mIOZlZNvy/CW/WJ0lVNHnyZKZOnbrJsYddhQEhSTvgt7/9LZdeeukmbf369eOOO+7YpG3ixIlMnLjlNcI33XRTNctrEQaEJO2A4cOHt4kzjarJg9SSpFIGhCSplAEhSSplQEiSSnmQWtJup6WvIWnO7cY7duzIoEGDyEw6duzItddey6c//ekWraO1GRCS1AL22msv5s6dC1ROgb3ssst46KGHalzVznGISZJa2F/+8he6d+8OwIMPPshJJ520cdkFF1yw8RqIiRMncsghhzB48GAuueSSWpS6Ve5BSFIL+Otf/8rQoUNZs2YNixcvZvbs2Vvtv3z5cu644w4WLFhARLBixYpWqrT53IOQpBawYYhpwYIF3HvvvZx11lls7V53++yzz8ab8/36179m7733bsVqm8eAkKQW9qlPfYqlS5eyZMkSOnXqxPr16zcuW7NmDQCdOnXiiSee4LTTTuPOO+/khBNOqFW578shJklqYQsWLGDdunX06NGDD3/4w7zwwgu88847rFmzhlmzZvGZz3yGt956i7fffpsTTzyRww8/nAMPPLDWZW/BgJC022nOaaktbcMxCKg8pW7atGl07NiRvn37Mnr0aAYPHkz//v059NBDAVi1ahUjR45kzZo1ZCZXX311q9e8LQaEJLWAdevWve+yK664giuuuGKL9ieeeKKaJe00j0FIkkoZEJKkUgaEJKmUASFJKmVASJJKGRCSpFJVPc01IhYBq4B1wNrMrI+I/YAZQB2wCBidmW9GRADXACcCbwNjM/OpatanTb3ynUG1LqHN+K/fnlfrErQTWvrf8o7+e6irq6OhoYGePXs2q/+KFSu45ZZb+PrXvw7AokWLePTRR/nSl74EVG78d9VVV3HPPffsUD3bqzX2II7OzKGZWV/MTwRmZWZ/YFYxDzAC6F+8xgFTW6E2SWozVqxYwY9//OON84sWLeKWW26pWT21GGIaCUwrpqcBpzRpn54VjwH7RsQBNahPkrbb6tWr+dznPseQIUMYOHAgM2bMAGDKlCkMGzaMQYMGsWDBAgAmTZrEVVddtXHdgQMHsmjRIiZOnMhLL73E0KFDmTBhAhMnTuThhx9m6NChW1xpvXr1as455xwOO+wwDj30UO66664W/07VDogE7ouIORExrmjbPzMXAxTvHyzaewOvNlm3sWiTpDbv3nvv5UMf+hDPPPMMzz333Mab7/Xs2ZOnnnqKr33ta5uEQpnJkyfz0Y9+lLlz53LllVcyefJkjjzySObOnctFF120Sd/LL7+cY445hieffJIHHniACRMmsHr16hb9TtUOiCMycxiV4aPzI+Lvt9I3Stq2uFduRIyLiIaIaFiyZElL1SlJO2XQoEHcf//9XHrppTz88MN069YNgFNPPRWAj3/84yxatKjFPu++++5j8uTJDB06lM9+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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6caa9be0>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='max_capacity',data=test,hue='car_type')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It can be seen that the training is fairly representative of the test set. This is important as we want our unseen scenerios to be similar to the seen scenerios to make our models as accurate as possible"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Another thing that is worth exploring is the travel times. Are they horly in nature? Are they just morning journeys? etc. To do this, we have to convert the travel times to meaning numerical data. One way will be to just extract the hour term. Another way is to convert is to minutes from midnight (another valuable insight provided by the community)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Extracting the hour term\n",
    "uber['hour_booked'] = pd.to_numeric(uber['travel_time'].str.extract(r'(^\\d*)').loc[:,0])\n",
    "test['hour_booked'] = pd.to_numeric(test['travel_time'].str.extract(r'(^\\d*)').loc[:,0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6cab21d0>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6c99a908>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "uber['hour_booked'].value_counts().plot.bar()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6cefccf8>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXoAAAD/CAYAAAD/qh1PAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAAELVJREFUeJzt3XuM5WV9x/H3h4tKhQDCQOkuuF5WhYSKdEXqLRS8gNpCE6hSIyvdum1ExWiiWzURq2kgaYtiDboRdTVWRJRCKLUgl1bSgi7KfbGsBNnNIqxysYi2Rb794zzbjsvszpndM3NmH96vZHJ+v+f3nOd8Z/a3n/PMc875TaoKSVK/dhp3AZKk2WXQS1LnDHpJ6pxBL0mdM+glqXMGvSR1zqCXpM4Z9JLUuaGCPsndSW5JcmOS1a3tGUmuSHJnu927tSfJOUnWJrk5yeGz+Q1IkrZuJjP636uqw6pqSdtfAVxZVYuBK9s+wHHA4va1HDh3VMVKkmZul+247/HAUW17FXAN8P7W/sUaXFvhuiR7JTmgqu7d0kD77rtvLVq0aDtKkaQnnxtuuOEnVTUxXb9hg76Ay5MU8JmqWgnsvym8q+reJPu1vguAdZPuu761bTHoFy1axOrVq4csRZIEkORHw/QbNuhfVlUbWphfkeSOrT32FG1PuHJakuUMlnY46KCDhixDkjRTQ63RV9WGdns/cBFwBHBfkgMA2u39rft64MBJd18IbJhizJVVtaSqlkxMTPubhyRpG00b9EmenmSPTdvAa4BbgUuApa3bUuDitn0JcEp7982RwMNbW5+XJM2uYZZu9gcuSrKp/99X1TeTfBe4IMky4B7gpNb/MuB1wFrgUeDUkVctSRratEFfVXcBL5yi/afAMVO0F3DaSKqTJG03PxkrSZ0z6CWpcwa9JHVuez4ZO6cWrfjHkY1195mvH9lYkjTfOaOXpM4Z9JLUOYNekjpn0EtS5wx6SeqcQS9JnTPoJalzBr0kdc6gl6TOGfSS1DmDXpI6Z9BLUucMeknqnEEvSZ0z6CWpcwa9JHXOoJekzhn0ktQ5g16SOmfQS1LnDHpJ6pxBL0mdM+glqXMGvSR1zqCXpM4Z9JLUOYNekjpn0EtS5wx6SeqcQS9JnRs66JPsnOT7SS5t+89Kcn2SO5N8NclTWvtT2/7adnzR7JQuSRrGTGb0pwNrJu2fBZxdVYuBB4FlrX0Z8GBVPRc4u/WTJI3JUEGfZCHweuCzbT/A0cCFrcsq4IS2fXzbpx0/pvWXJI3BsDP6jwPvAx5v+/sAD1XVY21/PbCgbS8A1gG04w+3/pKkMZg26JO8Abi/qm6Y3DxF1xri2ORxlydZnWT1xo0bhypWkjRzw8zoXwb8QZK7gfMZLNl8HNgryS6tz0JgQ9teDxwI0I7vCTyw+aBVtbKqllTVkomJie36JiRJWzZt0FfVX1TVwqpaBLwJuKqq3gxcDZzYui0FLm7bl7R92vGrquoJM3pJ0tzYnvfRvx94T5K1DNbgz2vt5wH7tPb3ACu2r0RJ0vbYZfou/6+qrgGuadt3AUdM0eeXwEkjqE2SNAJ+MlaSOmfQS1LnDHpJ6pxBL0mdM+glqXMGvSR1zqCXpM4Z9JLUOYNekjpn0EtS5wx6SeqcQS9JnTPoJalzBr0kdc6gl6TOGfSS1DmDXpI6Z9BLUucMeknqnEEvSZ0z6CWpcwa9JHXOoJekzhn0ktQ5g16SOmfQS1LnDHpJ6pxBL0mdM+glqXMGvSR1zqCXpM4Z9JLUOYNekjpn0EtS56YN+iRPS/KdJDcluS3JR1r7s5Jcn+TOJF9N8pTW/tS2v7YdXzS734IkaWuGmdH/F3B0Vb0QOAw4NsmRwFnA2VW1GHgQWNb6LwMerKrnAme3fpKkMZk26Gvgkba7a/sq4Gjgwta+CjihbR/f9mnHj0mSkVUsSZqRodbok+yc5EbgfuAK4IfAQ1X1WOuyHljQthcA6wDa8YeBfUZZtCRpeEMFfVX9qqoOAxYCRwAHT9Wt3U41e6/NG5IsT7I6yeqNGzcOW68kaYZm9K6bqnoIuAY4EtgryS7t0EJgQ9teDxwI0I7vCTwwxVgrq2pJVS2ZmJjYtuolSdMa5l03E0n2atu7Aa8C1gBXAye2bkuBi9v2JW2fdvyqqnrCjF6SNDd2mb4LBwCrkuzM4Inhgqq6NMntwPlJPgZ8Hziv9T8P+FKStQxm8m+ahbolSUOaNuir6mbgRVO038VgvX7z9l8CJ42kOknSdvOTsZLUOYNekjpn0EtS5wx6SeqcQS9JnTPoJalzBr0kdc6gl6TOGfSS1DmDXpI6Z9BLUucMeknqnEEvSZ0z6CWpcwa9JHXOoJekzhn0ktQ5g16SOmfQS1LnDHpJ6pxBL0mdM+glqXMGvSR1zqCXpM4Z9JLUOYNekjpn0EtS5wx6SeqcQS9JnTPoJalzBr0kdc6gl6TOGfSS1DmDXpI6N23QJzkwydVJ1iS5Lcnprf0ZSa5Icme73bu1J8k5SdYmuTnJ4bP9TUiStmyYGf1jwHur6mDgSOC0JIcAK4Arq2oxcGXbBzgOWNy+lgPnjrxqSdLQpg36qrq3qr7Xtv8TWAMsAI4HVrVuq4AT2vbxwBdr4DpgryQHjLxySdJQZrRGn2QR8CLgemD/qroXBk8GwH6t2wJg3aS7rW9tkqQxGDrok+wOfB14d1X9bGtdp2irKcZbnmR1ktUbN24ctgxJ0gwNFfRJdmUQ8l+uqm+05vs2Lcm02/tb+3rgwEl3Xwhs2HzMqlpZVUuqasnExMS21i9JmsYw77oJcB6wpqr+dtKhS4ClbXspcPGk9lPau2+OBB7etMQjSZp7uwzR52XAW4BbktzY2j4AnAlckGQZcA9wUjt2GfA6YC3wKHDqSCuWJM3ItEFfVdcy9bo7wDFT9C/gtO2sS5I0In4yVpI6Z9BLUucMeknqnEEvSZ0z6CWpcwa9JHXOoJekzhn0ktQ5g16SOmfQS1LnDHpJ6pxBL0mdM+glqXMGvSR1bpjr0WtrzthzROM8PJpxJGkzzuglqXMGvSR1zqCXpM4Z9JLUOYNekjpn0EtS5wx6SeqcQS9JnTPoJalzBr0kdc6gl6TOGfSS1DmDXpI6Z9BLUucMeknqnEEvSZ0z6CWpcwa9JHXOoJekzk0b9Ek+l+T+JLdOantGkiuS3Nlu927tSXJOkrVJbk5y+GwWL0ma3jAz+i8Ax27WtgK4sqoWA1e2fYDjgMXtazlw7mjKlCRtq2mDvqr+FXhgs+bjgVVtexVwwqT2L9bAdcBeSQ4YVbGSpJnb1jX6/avqXoB2u19rXwCsm9RvfWuTJI3JqF+MzRRtNWXHZHmS1UlWb9y4ccRlSJI22dagv2/Tkky7vb+1rwcOnNRvIbBhqgGqamVVLamqJRMTE9tYhiRpOrts4/0uAZYCZ7bbiye1vyPJ+cBLgIc3LfFo7hy66tCRjHPL0ltGMo6k8Zo26JN8BTgK2DfJeuDDDAL+giTLgHuAk1r3y4DXAWuBR4FTZ6FmSdIMTBv0VXXyFg4dM0XfAk7b3qIkSaPjJ2MlqXMGvSR1bltfjJVmZM0LDh7ZWAffsWZkY0lPBga9nrQ+9edXjWys0z599MjGkkbNpRtJ6pxBL0mdM+glqXMGvSR1zhdjpXnkb974hpGN9d6vXjqysbRjc0YvSZ0z6CWpcwa9JHXOoJekzhn0ktQ5g16SOmfQS1LnfB+9pGmtX/HtkYyz8MxXjGQczYxBL2mHdMYZZ8yrceYzl24kqXMGvSR1zqUbSRqRK696zsjGOuboH45sLGf0ktQ5g16SOmfQS1LnDHpJ6pxBL0mdM+glqXMGvSR1zqCXpM4Z9JLUOYNekjpn0EtS5wx6SeqcQS9JnZuVoE9ybJIfJFmbZMVsPIYkaTgjD/okOwOfAo4DDgFOTnLIqB9HkjSc2ZjRHwGsraq7quq/gfOB42fhcSRJQ5iNoF8ArJu0v761SZLGIFU12gGTk4DXVtWftv23AEdU1Ts367ccWN52nw/8YEQl7Av8ZERjjYo1Dceahjcf67Km4YyypmdW1cR0nWbjTwmuBw6ctL8Q2LB5p6paCawc9YMnWV1VS0Y97vawpuFY0/DmY13WNJxx1DQbSzffBRYneVaSpwBvAi6ZhceRJA1h5DP6qnosyTuAfwZ2Bj5XVbeN+nEkScOZjaUbquoy4LLZGHsII18OGgFrGo41DW8+1mVNw5nzmkb+YqwkaX7xEgiS1DmDXpI6Nytr9NJ0Jr0ja0NVfSvJHwMvBdYAK6vqf8ZaoNSRHXaNPsm7gIuqat20nZ/kkrwEWFNVP0uyG7ACOBy4Hfirqnp4DDV9mcFE4zeAh4DdgW8AxzA4L5fOdU3zVZLnAH/I4PMpjwF3Al8Zx7/bjiTJyxlckuXWqrp83PWM0468dPNR4Pok307y9iTTfjrsSexzwKNt+xPAnsBZre3zY6rp0Kp6I4MAew1wYlV9CTgVeNE4Ckrym0nOTfKpJPskOSPJLUkuSHLAmGp6F/Bp4GnAi4HdGAT+vyc5ahw1zVdJvjNp+23A3wF7AB9+sl9Fd0cO+rsYfOr2o8DvALcn+WaSpUn2GFdRSfZMcmaSO5L8tH2taW17jamsnarqsba9pKreXVXXVtVHgGePq6a2fLMHg1n9nq39qcCuY6rpCwx+y1kHXA38Ang98G0GYTsObwOOraqPAa8CDqmqDwLHAmePqaatSvJPY3royefNcuDV7Rx/DfDmcRSUZPckf5nktiQPJ9mY5Lokb53LOnbkNfqqqseBy4HLk+zK4NLIJwN/DYxrhn8BcBVwVFX9GAYzRWAp8DXg1WOo6dYkp1bV54GbkiypqtVJngeMay38POAOBh+q+yDwtSR3AUcyuOLpOOxfVZ8ESPL2qjqrtX8yybIx1QSD/6e/YvAkuAdAVd3TzvmxSHL4lg4Bh81lLZPslGRvBhPYVNVGgKr6eZLHtn7XWfNl4CLgtcAfAU9ncH5/KMnzquoDc1HEjrxG//2qmvJX/CS7VdUv5rqm9tg/qKrnz/TYLNe0J4Mlm1cwuJjS4QxmreuAd1XVTXNdU6vrtwCqakP7bedVwD1V9Z2t33PW6rmpql7Ytj9WVR+adOyWqjp0DDWdDiwDrgNeCZxVVZ9vS5Vfr6pXznVNra5fAf/CINg3d2RV7TbHJZHkbuBxBjUV8NKq+nGS3YFrq2rOn4Amn1Nt/7tV9eIkOwG3V9UL5qKOHXlG/8YtHRhXyDc/SvI+YFVV3QeQZH/grfz65ZvnTHvR7q1tSevZDP7d12+qb1yqasOk7YeAC8dYDsDFSXavqkc2C/nnMrqrq85IVX0iybeAg4G/rao7WvtGBsE/LmuAP6uqOzc/kGRc5/miLRx6nMFrQePw8yQvr6prk/w+8ABAVT2eZKonyVmxw87o56v2q+MKBn9sZb/WfB+DC7udWVUPjqs2TS/JCxj8/YTrq+qRSe3HVtU3x1fZ/JLkROCWqnrCE2CSE6rqH8ZQ1ryT5LeBzwLPA24F/qSq/qP9RnZyVZ0zJ3UY9HNn0jq55qEk7wTewWC2ehhwelVd3I59r6q2tC6tSTzPhzOXPyeDfg4luaeqDhp3HZpakluA362qR5IsYrCU9KW2fLLF14T06zzPhzOXP6cdeY1+Xkpy85YOAfvPZS2asZ03LddU1d3tfeoXJnkmU7/o+KTleT6c+fJzMuhHb38Gb6XafC0+wL/NfTmagR8nOayqbgRoM/s3MPjA2Zy/42ae8zwfzrz4ORn0o3cpsPumsJgsyTVzX45m4BQGlxj4P+2DZqck+cx4Spq3PM+HMy9+Tq7RS1LnduRLIEiShmDQS1LnDHpJ6pxBL0mdM+glqXP/C/PXmYcE4U7aAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6cab8a58>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "test['hour_booked'].value_counts().plot.bar()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "#express travel time in minutes from midnight\n",
    "test[\"travel_time\"] = test[\"travel_time\"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))\n",
    "uber[\"travel_time\"] = uber[\"travel_time\"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6d1214a8>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6ca01908>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "(uber[\"travel_time\"]/60).plot.hist(bins=100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6d044c88>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6d1213c8>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "(uber[\"travel_time\"]/60).plot.hist(bins=100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6d1b5f28>"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6d019c50>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "(test[\"travel_time\"]/60).plot.hist(bins=100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Both methods gave us almost the same information about the data. The most frequent travel time is around 7am and most of the journeys take place before noon with some journeys at 7pm and 11pm.\n",
    "\n",
    "Another column to explore is travel from i.e where do most of our customers come from?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6d315d68>"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYAAAAEqCAYAAAAcQIc3AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAAIABJREFUeJzt3XmcXFWZ//HPN2GTPUBYJGAAg4ACESObiCwOO4IjKIiYQRxwDIIOOoP6G0GQ+Tk6yCAiAhKMKCAKjEFQyAQUGGQJENbAEBYlsoVFQFFG8Jk/zqnkdlPdXXXrVnd17vf9evWru07dOn2qu6qee895zjmKCMzMrH7GjHQDzMxsZDgAmJnVlAOAmVlNOQCYmdWUA4CZWU05AJiZ1ZQDgJlZTTkAmJnVlAOAmVlNLTXSDRjMGmusERMnThzpZpiZjSq33XbbMxExfqjjejoATJw4kTlz5ox0M8zMRhVJv2nlOHcBmZnVlAOAmVlNOQCYmdWUA4CZWU05AJiZ1ZQDgJlZTQ0ZACQtJ+kWSXdKulfSl3P5BpJulvSgpB9JWiaXL5tvz8/3TyzU9flc/oCk3bv1pMzMbGitXAG8AuwSEVsCk4E9JG0L/BtwakRMAp4HDs/HHw48HxFvBk7NxyFpM+Ag4K3AHsC3JY2t8smYmVnrhpwIFmnT4D/km0vnrwB2AT6cy2cAJwBnAvvlnwF+AnxLknL5RRHxCvCIpPnA1sCv22nwxOOuaOm4R7+6dzvVmpnVTktjAJLGSpoLPA3MAh4Cfh8Rr+ZDFgDr5p/XBR4DyPe/AKxeLG/yGDMzG2YtBYCIeC0iJgMTSGftmzY7LH/XAPcNVN6HpCMkzZE0Z+HCha00z8zMSmgrCygifg/8EtgWWFVSowtpAvB4/nkBsB5Avn8V4LlieZPHFH/H2RExJSKmjB8/5FpGZmZWUitZQOMlrZp/fgPwXmAecC1wQD5sKvDT/PPMfJt8/zV5HGEmcFDOEtoAmATcUtUTMTOz9rSyGug6wIycsTMGuDgifibpPuAiSV8B7gDOzcefC5yfB3mfI2X+EBH3SroYuA94FZgWEa9V+3TMzKxVrWQB3QW8vUn5w6TxgP7lfwYOHKCuk4GT22+mmZlVzTOBzcxqygHAzKymHADMzGrKAcDMrKYcAMzMasoBwMysphwAzMxqygHAzKymHADMzGrKAcDMrKYcAMzMasoBwMysphwAzMxqygHAzKymHADMzGrKAcDMrKYcAMzMasoBwMysphwAzMxqygHAzKymHADMzGrKAcDMrKYcAMzMasoBwMysphwAzMxqygHAzKymhgwAktaTdK2keZLulXRMLj9B0u8kzc1fexUe83lJ8yU9IGn3QvkeuWy+pOO685TMzKwVS7VwzKvAsRFxu6SVgNskzcr3nRoR/148WNJmwEHAW4E3Av8laeN89xnA3wALgFslzYyI+6p4ImZm1p4hA0BEPAE8kX9+SdI8YN1BHrIfcFFEvAI8Imk+sHW+b35EPAwg6aJ8rAOAmdkIaGsMQNJE4O3AzbnoKEl3SZouaVwuWxd4rPCwBblsoPL+v+MISXMkzVm4cGE7zTMzsza0HAAkrQhcAnw6Il4EzgQ2AiaTrhBOaRza5OExSHnfgoizI2JKREwZP358q80zM7M2tTIGgKSlSR/+P4yISwEi4qnC/ecAP8s3FwDrFR4+AXg8/zxQuZmZDbNWsoAEnAvMi4hvFMrXKRz2fuCe/PNM4CBJy0raAJgE3ALcCkyStIGkZUgDxTOreRpmZtauVq4A3gUcCtwtaW4u+wJwsKTJpG6cR4EjASLiXkkXkwZ3XwWmRcRrAJKOAq4CxgLTI+LeCp+LmZm1oZUsoBto3n9/5SCPORk4uUn5lYM9zszMho9nApuZ1ZQDgJlZTTkAmJnVlAOAmVlNOQCYmdWUA4CZWU05AJiZ1ZQDgJlZTTkAmJnVlAOAmVlNOQCYmdWUA4CZWU05AJiZ1ZQDgJlZTTkAmJnVlAOAmVlNOQCYmdWUA4CZWU05AJiZ1ZQDgJlZTTkAmJnVlAOAmVlNOQCYmdWUA4CZWU05AJiZ1dSQAUDSepKulTRP0r2Sjsnlq0maJenB/H1cLpekb0qaL+kuSVsV6pqaj39Q0tTuPS0zMxtKK1cArwLHRsSmwLbANEmbAccBsyNiEjA73wbYE5iUv44AzoQUMIDjgW2ArYHjG0HDzMyG35ABICKeiIjb888vAfOAdYH9gBn5sBnA/vnn/YDvR3ITsKqkdYDdgVkR8VxEPA/MAvao9NmYmVnL2hoDkDQReDtwM7BWRDwBKUgAa+bD1gUeKzxsQS4bqLz/7zhC0hxJcxYuXNhO88zMrA0tBwBJKwKXAJ+OiBcHO7RJWQxS3rcg4uyImBIRU8aPH99q88zMrE0tBQBJS5M+/H8YEZfm4qdy1w75+9O5fAGwXuHhE4DHByk3M7MR0EoWkIBzgXkR8Y3CXTOBRibPVOCnhfKP5mygbYEXchfRVcBuksblwd/dcpmZmY2ApVo45l3AocDdkubmsi8AXwUulnQ48FvgwHzflcBewHzgZeAwgIh4TtJJwK35uBMj4rlKnoWZmbVtyAAQETfQvP8eYNcmxwcwbYC6pgPT22mgmZl1h2cCm5nVlAOAmVlNOQCYmdWUA4CZWU05AJiZ1ZQDgJlZTTkAmJnVlAOAmVlNOQCYmdWUA4CZWU05AJiZ1ZQDgJlZTTkAmJnVlAOAmVlNOQCYmdWUA4CZWU05AJiZ1ZQDgJlZTTkAmJnVlAOAmVlNOQCYmdWUA4CZWU05AJiZ1ZQDgJlZTTkAmJnV1JABQNJ0SU9LuqdQdoKk30mam7/2Ktz3eUnzJT0gafdC+R65bL6k46p/KmZm1o5WrgC+B+zRpPzUiJicv64EkLQZcBDw1vyYb0saK2kscAawJ7AZcHA+1szMRshSQx0QEddJmthiffsBF0XEK8AjkuYDW+f75kfEwwCSLsrH3td2i83MrBKdjAEcJemu3EU0LpetCzxWOGZBLhuo3MzMRkjZAHAmsBEwGXgCOCWXq8mxMUj560g6QtIcSXMWLlxYsnlmZjaUUgEgIp6KiNci4q/AOSzu5lkArFc4dALw+CDlzeo+OyKmRMSU8ePHl2memZm1oFQAkLRO4eb7gUaG0EzgIEnLStoAmATcAtwKTJK0gaRlSAPFM8s328zMOjXkILCkC4GdgDUkLQCOB3aSNJnUjfMocCRARNwr6WLS4O6rwLSIeC3XcxRwFTAWmB4R91b+bMzMrGWtZAEd3KT43EGOPxk4uUn5lcCVbbXOzMy6xjOBzcxqygHAzKymHADMzGrKAcDMrKYcAMzMasoBwMysphwAzMxqygHAzKymHADMzGrKAcDMrKYcAMzMasoBwMysphwAzMxqygHAzKymHADMzGrKAcDMrKYcAMzMasoBwMysphwAzMxqygHAzKymHADMzGrKAcDMrKYcAMzMasoBwMysphwAzMxqygHAzKymhgwAkqZLelrSPYWy1STNkvRg/j4ul0vSNyXNl3SXpK0Kj5maj39Q0tTuPB0zM2tVK1cA3wP26Fd2HDA7IiYBs/NtgD2BSfnrCOBMSAEDOB7YBtgaOL4RNMzMbGQMGQAi4jrguX7F+wEz8s8zgP0L5d+P5CZgVUnrALsDsyLiuYh4HpjF64OKmZkNo7JjAGtFxBMA+fuauXxd4LHCcQty2UDlryPpCElzJM1ZuHBhyeaZmdlQqh4EVpOyGKT89YURZ0fElIiYMn78+EobZ2Zmi5UNAE/lrh3y96dz+QJgvcJxE4DHByk3M7MRUjYAzAQamTxTgZ8Wyj+as4G2BV7IXURXAbtJGpcHf3fLZWZmNkKWGuoASRcCOwFrSFpAyub5KnCxpMOB3wIH5sOvBPYC5gMvA4cBRMRzkk4Cbs3HnRgR/QeWzcxsGA0ZACLi4AHu2rXJsQFMG6Ce6cD0tlpnZmZd45nAZmY15QBgZlZTDgBmZjXlAGBmVlMOAGZmNeUAYGZWUw4AZmY15QBgZlZTDgBmZjXlAGBmVlMOAGZmNeUAYGZWU0MuBrfEO2GVFo97obvtMDMbZr4CMDOrKQcAM7OacgAwM6spBwAzs5pyADAzqykHADOzmnIAMDOrKQcAM7OacgAwM6spBwAzs5pyADAzqymvBVSxzWds3tJxd0+9u8stMTMbXEdXAJIelXS3pLmS5uSy1STNkvRg/j4ul0vSNyXNl3SXpK2qeAJmZlZOFV1AO0fE5IiYkm8fB8yOiEnA7HwbYE9gUv46Ajizgt9tZmYldWMMYD9gRv55BrB/ofz7kdwErCppnS78fjMza0GnASCAqyXdJumIXLZWRDwBkL+vmcvXBR4rPHZBLutD0hGS5kias3Dhwg6bZ2ZmA+l0EPhdEfG4pDWBWZLuH+RYNSmL1xVEnA2cDTBlypTX3W9mZtXo6AogIh7P358GLgO2Bp5qdO3k70/nwxcA6xUePgF4vJPfb2Zm5ZUOAJJWkLRS42dgN+AeYCYwNR82Ffhp/nkm8NGcDbQt8EKjq8jMzIZfJ11AawGXSWrUc0FE/ELSrcDFkg4HfgscmI+/EtgLmA+8DBzWwe82M7MOlQ4AEfEwsGWT8meBXZuUBzCt7O8zM7NqeSkIM7OacgAwM6sprwXUw+ZtsmlLx216/7wut8TMlkS+AjAzqylfAdTIGZ+4pqXjpn1nly63xMx6ga8AzMxqygHAzKym3AVkpZ3yoX1aOu7YH/2syy0xszJ8BWBmVlMOAGZmNeUAYGZWUw4AZmY15QBgZlZTDgBmZjXlAGBmVlOeB2A9Y8Fx1w95zISvvruluk444YRKjzNbEjkAmLVg9jUbtXTcrrs81OWWmFXHAcBsBKx97dwhj3ly58nD0BKrM48BmJnVlAOAmVlNOQCYmdWUA4CZWU05AJiZ1ZQDgJlZTTkAmJnV1LDPA5C0B3AaMBb4bkR8dbjbYLYkmXjcFS0d9+hX9+5yS2y0GdYrAEljgTOAPYHNgIMlbTacbTAzs2S4rwC2BuZHxMMAki4C9gPuG+Z2mNkAKr+iOGGVFo97YchDNp+xeUtV3T317paOm7fJpi0dt+n981o6brRRRAzfL5MOAPaIiI/n24cC20TEUYVjjgCOyDffAjzQQtVrAM9U2NRerq+X21Z1fb3ctl6vr5fbVnV9vdy2qutrta43RcT4oQ4a7isANSnrE4Ei4mzg7LYqleZExJROGjZa6uvltlVdXy+3rdfr6+W2VV1fL7et6vqqbttwZwEtANYr3J4APD7MbTAzM4Y/ANwKTJK0gaRlgIOAmcPcBjMzY5i7gCLiVUlHAVeR0kCnR8S9FVTdVpfRKK+vl9tWdX293LZer6+X21Z1fb3ctqrrq7RtwzoIbGZmvcMzgc3MasoBwMysphwAzMxqygGgZiTNkTRN0riRbouZDU7SPpK69jntQeB+JK0ERET8YaTb0g2S3gwcBnwImAOcB1wdS/gLQdIKEfHHCupZC3hnvnlLRDzdaZ2dkrRJRNwvaatm90fE7cPdpm6T9NFm5RHx/Tbr2SUirpH0twPUd2mZ9lVF0g+A7YBLgPMiotI1KUZdAJD0kYj4gaR/bHZ/RHyjZL2bA98HViPNWF4ITI2Ie9qs5+KI+KCku+k7y1mpebFFG3XdEBE7SHppgLpWbqdt/eoeA+wDnAn8FZgOnBYRz5Woazzwz6QF/pZrlEfELiXqWgv4V+CNEbFnXixwu4g4t926cn3bA98FVoyI9SVtCRwZEZ8sUdcHga8DvyT9D94NfC4iflKmbbnO5YDDgbfS92/3sTbqOCci/l7StU3ujjL/h0Ldezdp24kl65oE/H9e/zrZsERdpxduLgfsCtweEQe0Wc+XI+J4Sec1uTva+T8U6vwa8BXgT8AvgC2BT0fED9qtK9e3MnAw6cQtSCdtF0bES2Xq6yMiRtUX6c0LcHyzrw7qvRHYuXB7J+DGEvWsk7+/qdnXSP/9ctu2AE4lrbP0TWAb4Fhgbsn6riZ9iM0D3kMKJv9Wsq6fAx8E7sy3lwLu7uC53kyafX5HoeyeknXdCaxZuD2+0c4O2vdj4CTgIWBq/lueNtKvkdy275BOih7L76+7gXM7qO8G0gf1Xfn9cALw5Yraugows+RjxwAfrPDvNjd/fz8wg3RS2enrZA3g08Cj+T3yIPCpjts60i+yXvlq9g8q+08jTXL7r4raNabsB9YA9d0GzAY+DCzb775Ly9aZv99VKPtVybpuzd+LH9ilAlN+7M1N6iv7f7273+0xnQSnYrsafztgaeCaNuv428G+OmjbXf2+r0jqLiz92uv/dwSu7+TvV6hnaWBeB4+/rop25Lruzd/PIS1+2clrbl/gshw0P0c+AQGWB37TaVuHfUOYTkn6p4j4Wr4EfF3/VUQcXbLqhyX9C3B+vv0R4JEyFUXEa5JelrRKRAy9xu3gdf1V0p2S1o+I33ZSV3Zg5OW4m/yupv2gLfhL/v5E7jJ4nLTOUxl/lLQ6+X8raVugk7/hY7kbKPLyI0eTrlTK+IWkq4AL8+0PAVd20DZY/Lf7vaS3AU8CE9usY9/8fU1ge+CafHtnUndV2X7sP+XvL0t6I/AssEHJugD+nLseH8wrAvyO1Oa2Sbqcxe//scCmwMUdtG2WpM8CPwIWjRVFiS5R4HJJ95P+fp/MXaR/LtmuA4FTI+K6YmFEvCyp7e6p/kbjGMC+EXG5pKnN7o+IGSXrHQd8GdiB1L97HXBCRDxfsr6LgW2BWfR9QbUdoCRdQxp4vKVfXe8r2bbK+nVzffsA15O6Wk4HViZd2re9zlMeyDwdeBtwD6mb5YCIuKtk29Yg7UD3XtL/9WrgmIh4tmR9HwDeleu6LiIuK1NPob6Pkwb4tiD17a4IfCkivlOirp8Bfx8RT+Tb6wBnlA3s+YTodFK3zRmkD9xzIuJLJet7Jyn4rkrq9loZ+FpE3FyirvcUbr5KOhteUKZdub5mJ3sRJcYncn3jgBfzyeDywMoR8WTZ9nXLqAsAzeSzihUj4sWRbktDlQGq34u9WNevStT1HdLl486kwdEDSNksh7dbV6HO1fqfKUnaICJKXUFJWoq0F4SAByLiL0M8ZKB6xgJHR8SpZR4/2ki6JyLeVrg9htR987ZBHtZq3csCy3VyRSvpwIj48VBlbdS3NmmTqSB1HY7oB2w3MoryFfDppCucZUhXO3+MDhJA+tQ/WgOApAuATwCvkfq1VwG+ERFfL1lf8ZKy4QVSquRZEVH2Eq4SVaUfSrorIrYofF+R1Pe/Wwdt+29gz0YAlrQp8OMyHzwDvHleIPUbt/2cJf0yInZq93H96uifhdVHJ2/G/MH6AVK3z6Iu2TJXZJK+BUwidVEFabXd+RHxqZJtu550JXw98N/RYdaJpNsjYquhylqs6+PAl0jdXSIlH5wYEdNLtm1p4B+AHXPRL0nv+5ZPPrqUUTSH9H/8MTAF+Cjw5oj4Yrt1Na1/FAeAuRExWdIhwDtIaYi3RRtplv3qO43U3VDs330SeAPp8u3QNut7hOZjFGVS3ipLP5R0c0RsI+km0iDhs6RB5knt1lWoc2/gn4C9SWfu3wcOiYi5Jeq6gpT33Ehp3Am4CdiY9AY/f4CHDlTfyaSTg/59u23nxks6kfSaOJ/0fzgEWCkivtZuXYU6f0EKcLeRTmYa7TulZH3vZ/GHWEddVJI2JHWJvpvUnfkKadD2M23WsyewFym760eFu1YGNouIrUu07QFg+0ZXXh43ujEi3tJuXfnx3yUNJDeu0A8FXou8e+FIUd4ApnHClstujIjtq6h/1A0CFyydo/b+wLci4i+SOolmb4+IHQu3L5d0XUTsKKnMktXFXXuWIw3mrFaybV8E3tk4A86DSv8FlMk//5mkVUkB5XZyv27JdgEQEVfk/8XVwErA/hHxYMnq/gpsGhFPwaIrnzNJqarXsXiQvlWNN0rxjDqAMrnxu0fENoXbZ0q6GSgdAIAJEbFHB4/v70ZSn3iQxoxKi4iHJf0J+N/8tTOpK6Jdj5OupN9HCnQNLwFtBZOCBfnxxboeK1kXpPfXloXb10i6s0xFaj5H6QXSCWq7J0Uv5+SFuXl+wRPACmXa1cxoDgBnkXJi7wSuk/QmoJMxgPHFTBtJ65NybyG9+NvSZJDxPyTdQLpsbdeYft0fz1JyGY+IOCn/eEkeNCzdr9skE2tl4GHgU5LKZmRNbHz4Z08DG0fEc5LaHguIiJ1LtGEgr+UrzotIz/tgCmftJd0oafOIaG0X80E0uVI8XVLpiWqSHiLtP3sBcC4p7/yv7dYTEXcCd0r6YUS8WqYthTY1Plx/B9ws6aek/8V+dBbwXpO0UUQ8lH/PhpT/307JX5fn23uTNsP6hKQft3nFeCjpvX4UKViuR+oyrMSoDQAR8U3SJKaG30jq5M1+LHBDftGLlO72SUkrsPiysGXqOy1/DOkFsVLJtlWSfpiD5B8j4pk8uLQDMB/4z5LtmtPv9m1Nj2rP9TkwNQYGP0AK8CsAvy9TYYVZTx8mZRSdRvrQ+e9cVqZNjZniSwGHSXqY1MXS9ozxgiqvFCG9v3YgBbq3A7/KV8UPtVOJ8ux44I5mV+ltPtfGe+ih/NXw03ba1MTngGvz/wHSmMxhJetaHdgq8nIyko4n/Q92JL1HWg4AEfGb/OOfSVmKlRp1YwDq0lIQue5lgU1Ib8L7Oxn4Vd9p+a+Srlb+PSIeKFlfR+mHOaXv70gfOheR0iJ/SepauTMiPl2mXYX6lyH100NnmTsifeg3nusNwCVR8oXajaynKuRgPKDCG7+dOu+OiM0Lt8eQ/rebD/KwVupdkfRh+FlSl9XYNh+/TkQ8MdBzLvNcq6KUmvpYRDyZ3/9Hkt4bTwLH9c9ua7HOecCWEfG/+faypMmMm0q6IyLe3kIdk0gB/TngG6Ru2neTgt7HI+LWdtvVzGi8Amj0fzU7m+40mr2DxdkYW+RujLYWl1rUkGq7HoiIS0j54mUdTOq/XR74LbB2pMkkSwFtD9YWSdqJdJX0KOlDez1JU6Pf5JVW5A/6n1D+rLW/7QtZT1+WdAolJ0blM+q/5/UZO21ndzQ+9PKV2L2NDBulxQg3A8p8KDa7Uvx5iXrIbTmFdAWwIvBrUvfl9e3WE3leQkT8RhWlbkqaQvqAfBN9/xftXjmdRfrAh3QydBzwKWAyafvFttYWyi4AbsrdU5Am6l2Yr2Lva7GO80jJFCuTljP5NGlpiXcD38pt7dhovAKYEANM+FCeJFay3vOBjUgfho2+vyjZj92os5KuBzVPQ2ykqB4bA8zs7VfHonS7/mchZVPxCo+/Dfhw4+pG0sakxareUaKuvwX+jTRDVNDZwneSbomIravIepJ0I+kDsH/GTunALOkOUndBY+bzGGBO2f9H/vstmszYYRbQgbmOp4Y8uLX6KkvdzFlAnyOtT7RoXKLdqwlJdzYGfyWdASyMiBPy7bkRMbndtuXHvoPF/4cbIqJ/d+lQj1/0uyXNj4g3N7uvU6PxCmC2pN0j4tFioaTDgP/H4oGXdk0hpaRVEhEH6nooWd03SJkUF5BeUAcBa5MWc5tOSpUcyqr5w0HAylqcby9SmmQnli52bUXE/+SsoDK+Buwb1S17e3mFWU/LR8Q/V9SuBhVfc5GW/ij9vow02ehSAEljJR0SET9sq0F5eWlSd8O6ktbt9zvKLi/9OVK2XZ/UTdJruF0Lo8RM8ybGSloqD07vChxRuK/U/yEnkCwkreGzqCzaW8qlONjeP7ml7YH4gYzGAPAZ0rodezVSDSV9njQY13TGbIvuIX2oPtF5E4EKux5IC0oVL/nOlnRTRJwo6Qst1vErFq8Zc13h58btTsyRdC6LUzQPofyA8FNVffjns+nZEfF7Ksh6IqXQ7hURna7/U/SwpKNJqa4AnyRlUrVMabngacC6wEzS8iPTSB+4c4G2AgDwj6QPwmZzEcqm0EK1qZvHK+XuzyYNnqfGtT/b9kLS4PYzpLV7rgdQ2jej7OvkChZfsb+BlFDyAKk3oFWbSLqLdIK2Uf6ZfLvU8hTNjLouIABJu5L67vYHPk6aIbtPlFy3J9d5Lanf7xb6vqDKrrdT2YQrSb8mLd/c6Bc/APjHiNi2ysvBsvIg1zT6rqP07Yh4ZdAHNq/rNFIg/k86e2M36vt1RGxX5rFN6nqJNAbVyIuvYl+GNUnZNruQPjRmk9aOb3nWc+5rfp7UT78rMI60bMAxUWIyXtUKCRuTgc1JGTuLUjcj4hMl6vwBKWHjXhafEUeZ8Zg8DrMOaaXTP+ayjUnLy3S8mY5SRuCREXFkG4+pPEmg6e8ZjQEAQNIOpA+JG0lreXe0VIMqXG8n19dsIa3vRsS/lKhrQ1Lq4Xa5nptIV0K/A94RETeUaWMVlNbbmRERH6movsqm0ef6vkxaSvfSqrr3ek0x+yf/P54B1o8KNgxRWkl1In0HWtvddev4we6PiLbTG/tnPPW6TsfZumXUBYDCgKiAZUnL6b5GNWdjXdnuTxUspNXLcubJvo20t15SOGt/lZRLXfp1Iqmx/MMGEXGSpPVIGwCVnoBURWZR/w+Xqj5supEYURVJ55CWSW41q2bYqG+K+hhgK2D1iNh9hJo0oFEXALpFFa23o8FXBAxSXu8NETHkLEN1b++DSkk6i/Qin0nf9XbanpORL73PBNaKiLdJ2gJ4X0R8par2liWpsX3mLpFyuseRug3eOcRDB6uz48wiSa+x+O8uUr/zy3SeQTWPChIjJA06WFummzW3bSPSnh2dTqCrVL8rnsYcoEs67aXohtE4CNwtVc2ifA8pzW3fAe5fnZSt9Dct1NUYDG0rhWwoShuP9N+XtdR8h+zx/DWG8rOdG84hDV6eldt1l9LKr20FAHVno/RtImKrnLpJRDyvNAGuEx1nFkWbE7PaUFVixHakwd4LSTnt6rA+gCrXT6raJdHmXuIjxQFgsUrW24mI4/P3AaeR54yZVuq6PH8vtcnNAL/7eFLa6Gak5ST2JM22LR0AGn24klZoDKJ1YPmIuCX1tixSZv2YZpksxTPZMpksf8l97I2c/fF0npLXjcyijmjx0ugrAfdJ6jSLYluCAAAJh0lEQVQxYm3SCc/BpGy9K0jzRMossrioGR08ttu+k08MvgdckLPQSlGFqwo34wCwWCWzKNXCUhXR4jIE3bh0JmUQbUnai/awPO7x3RL1LCJpO9JiYSsC60vakpT18MkS1T0jaSMWf8geQLkz0O9KWjvyjGylDXo+QLocP6FEfZCydS4D1lRaZvoA0tVcJ44BviDpFdJ4VsdjWRX49yory92dvyC9x5YlBYJfSjoxIk4vWW0j1VKkK9kyqZZdERE75K7Mw0gp0rcA50XErBLVVbmq8Ot4DKBAFcyilHRkRJw1QOZDRBszgSUtZJBL5zIZSlo8M/Y20iS1l0jpqaXfOEpLIh8AzIw8w1j9dqdqo64NSVPwtyelNj4CfCT6TfxroZ7bgfdGWkV0R9L6R40p/ptGRJkp/kjahJTZJdIcg6omrC3R8gf/3qQP/4mk8aLpEfG7iupvO9Wy2/LV4v6kE4cXSa+ZL5RNaS7Ue0NE7FBBE30FUBQVzKIknZk0TW2TNNC4wEC6cek8R2lm7Dmkgcc/0OG68QAR8Vi/bptSS+lGWtbivUrrpozpIJVxbCxeyOtDwNl5YPUSSaVy45U2hLke+F4FXV3FeseRdvIqjsl0Ojmvk/ZUugOapBmkPZ5/TtoruvL+8Yi4XWlhtxGXExcOIwW8WaQMudslvZE0V6PlAKBqVxV+ndoHAFU/i7KypSq6celc6Jb5jtJuVCtHyQ3XCx7L+eKR+z6PZvEAdlvUb4vERlBp58opq3yKP6n76GDgm/lD8nrSlWLppYiV1sc5BphAer1tS/qQKDvbtmMRsVJuW9Md0EpUeSgpS2lj4OjCiUInKbnNUi0XlmhbN5wD/Iz0On6wkf0TEY9LarfLsDiG1cgo+mAVjQR3AVU+i1LSXqRJW82WqtgzBljIbpD6Kr90zmcoE+mbd176slTSGqTn/F7Sm/pq0mbsZZbSrWSLRElfJG1D+AywPnnBNaUp/jMi4l3ttq1Q99qkN+FngXGND8ySdd1NmntyU6QtTjchnSV/qGydVVGezT5U2UjoxVRLpTWc/hX4GGnFXZEC+3nAF6PkEund5ADQhVmUqmipin6XzhdVceksaTqwBdVMoa98ZdayYwcD1FXpFH+ltWc2A54inf3fANweHexyJenWiHhn7pbaJiJeUQ8s75HbdiNpFntxB7RpUdF+tFWoKPOsEpJOJV0hfSYWL++9MmlQ/U8RcUwbdTVNImmIDvY9Kap9FxAp8wJIXS6SHunkwz/XM1vS35Emld0I7FryzKTyS2dg24jYrMTjmunGyqyVbZEYETc1KfufDqpcHRhL2pnsOeCZTj78swV5TOY/SYscPk+aU9ELKtsBrWoVZ55VZR/S9qXF1V1flPQPwP2krr5WNa4q30I6gWxkBO5L54s3LuIrgIpnUaqLS1VUIc9BOCUqmEJfZXeX+m6ROIm0ImZPzfBskLQpsDtpPaaxETGhonrfQ1qa+xfRg8tq9JIqM88qbNP/RMTG7d43RJ1XAx+IvhsG/TgiKpkIV/srgKh4FmUn/cHDZAbwa0lP0uEHbERcmfPXfy6p2N21Y7vdXaSzp54maR/SEiE7ksaKrqGCs7Hc9bgWKeUVUvZXO2vHV0rSlwa5OyLipGFrzCCqyjyr0H2SPhr9ZtVL+gjpCqCM9Ukrzzb8L2n8rhK1DwA1NJ3UtdRnJ6Wyquruiu5skVi1PUkf+KfljI7GhunNVjBtiaRPAceTxhUWjcmQxmlGSrM+9RWAw0ndYL0QACrLPKvQNOBSSR8jJTEE6YToDaTtHMs4H7hF0mW5vvfTwaz9/mrfBVQ3kq6JiEpSDLvR3aWKt0ismqTJpC6uD5LO2C+JiG91UN980uDvsxU1sVI5AB9D+vC/mNR9WMkquZ2oMvOsapJ2Ic1IFulkZnaH9TW2l4SUdnxHh01cxFcA9XO/0uJql9Phhitd6u6qdIvEShqUMocOIp3tPwv8iNTOnSuo/jHK7zzVNZJWI62ndAip23CrEt16lWtknkXEM6S2Fe/bl/JbwlYmIq4hdQ9WZS5pOZSlANT+9pIDcgConzeQPvh3K5QF5berrFrHWyR2wf2ktM99I2I+gKTPdFJhIc3vYdLkvivoG5ArSfMrQ9LXSbvYnQ1sHhF/GKm2NNGtPcF7Ur8uwkVX11TUReguIOspqmCLxC606f2kK4DtSTOzLyLt7rZBB3VWvktWVST9lRSMXqXvkhAjnslW9UTLXtftLkIHgJqRNIG0VeW7SG/uG0iznpeoN043KK1PtD+pK2gXUtfIZRFxdRV198qEpl5X1UTL0UBpr/K/qWC+SfP6HQDqRdIs4AJSdgHAR4BDIqKVDWq6TtJypAHHt9J3cbRSewJ3S+4jPxD4UCeD6sUJTRHRKxOaep4q3hO8V+V5O28hLQRZeRdh2xue2Kg3PiLOi4hX89f3gPEj3aiC80l58LsDvyKtpdLx5uZVi4jnIuKsCjKq/oP0XJ/N9d5JmmdgTUh6SdKLpOVRViat3/V0oXxJ81vSApXLkGYHN74q4UHg+nkmT0xpbHzTyGzpFW+OiAMl7RcRM3LG0lUj3ahu6sEJTT1rFEy0rFRUu9ve6/gKoH4+Rsphf5KUWnZALusVjbWZfq+0d/EqVDjzsQf1mdAk6bOM/IQm6xGStpN0H/k1IWlLSd+urH6PAVgvUVof/xJSmtt5pMW+/iUizhrRhnVJL09ospHX7TWPHABqQtLpDL7L09HD2JwBSRobaSOcJVo3ltK2JY/y/guS7igEgDsjYssq6ncXUH3MIa1PchvwvsLPja9eMV/S1yVVtWR1r5otaWL/wjyh6T+GvTXWq7raRegrgBoqnk30mrz2zEGkPVXHkBavuygilqgMj7pNaLJyut1F6ABQQ5Ju75XF1QYjaUdSttKqwE+AkxpLMSwJ6jShydozXF2E7gKyniJprKT35eVvTyNtir0haY2XK0e0cRXLq0T+HWkp7Q1JS2n7w99gmLoIPQ+gJgpLNwMsX5g0M+Lru/TzIHAt8PWIuLFQ/pN8RbBEaLKUdmNCU6/9P2xkfIa0RWizLsL3VPVL3AVkPUXSij22+qTZiBiOLkIHAOsJo2UbQrPh1O01jxwArCdIOrZJ8fKkM5/VI2LFYW6S2Yjpxm57TX+PA4D1ml7dhtBsSeNBYOsZvboNodmSygHAekKPb0NotkRyF5D1hF7ehtBsSeUAYGZWU54JbGZWUw4AZmY15QBgZlZTDgBmZjX1f2w4Kw0pITBcAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6d280518>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "uber['travel_from'].value_counts().plot.bar()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6d446128>"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6d4523c8>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "test['travel_from'].value_counts().plot.bar()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "So many of our customers are actually coming from Kisii. The training set is fairly representative of the test set too.\n",
    "\n",
    "We can also explore to see if people are likely to travel on a particular day of the week more than the rest."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "uber[\"travel_day\"] = uber[\"travel_date\"].dt.day_name()\n",
    "test[\"travel_day\"] = test[\"travel_date\"].dt.day_name()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "uber[\"travel_yr\"] = uber[\"travel_date\"].dt.year\n",
    "test[\"travel_yr\"] = test[\"travel_date\"].dt.year"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Calculating the number of weeks in the dataset we have data for\n",
    "a=uber[uber[\"travel_yr\"]==2018][\"travel_date\"].dt.week.nunique() + uber[uber[\"travel_yr\"]==2017][\"travel_date\"].dt.week.nunique()\n",
    "b=test[test[\"travel_yr\"]==2018][\"travel_date\"].dt.week.nunique() + test[test[\"travel_yr\"]==2017][\"travel_date\"].dt.week.nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6d5880f0>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6d581b00>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "(uber[uber['car_type']=='shuttle'][\"travel_day\"].value_counts()/a).plot.bar()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6d58b5c0>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXQAAAEsCAYAAADTvkjJAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAAF4pJREFUeJzt3XuYZVV55/Hva6NiBBSkVAzBxthyiQiYlqgEAuKFCIoGL4OP2EFMmxkwxjiZMGYSJGMyOA4m0cdH09ooMQpi0GjiBRhCuCheios0DigMohIaaUJUYhwReOePtQ9dtFVd51TVqX3W6u/neeqp2rtOnfPS7PPb66y99lqRmUiS6veQvguQJC0NA12SGmGgS1IjDHRJaoSBLkmNMNAlqREGuiQ1wkCXpEYY6JLUiO2W88V23XXXXLly5XK+pCRV78orr7wzM6fme9yyBvrKlSuZnp5ezpeUpOpFxLeHeZxdLpLUCANdkhphoEtSIwx0SWqEgS5JjTDQJakRBrokNcJAl6RGLOuNRaNaecpnxvr8t5x+1FifX5KWky10SWrERLfQq/fWR435+X8w3ueXVBVb6JLUCANdkhphl4vmtN9Z+43tuTes2TC25wa4fu99xvr8+9xw/VifX1oIW+iS1AgDXZIaYaBLUiMMdElqhIEuSY0w0CWpEQa6JDXCQJekRhjoktQIA12SGmGgS1IjDHRJaoSTc0kT6D2//Y9jff6T3vecsT6/+mELXZIaYaBLUiMMdElqxLyBHhG/EBEXR8T1EfH1iHhjt3+XiLgwIm7svu88/nIlSXMZpoV+L/DmzNwHeCZwUkTsC5wCXJSZq4CLum1JUk/mDfTM3JiZV3U/3w1cD/w8cAxwVvews4CXjKtISdL8RupDj4iVwIHAl4HHZeZGKKEPPHaOv1kbEdMRMb1p06bFVStJmtPQgR4ROwDnAb+bmT8c9u8yc11mrs7M1VNTUwupUZI0hKECPSIeSgnzj2TmJ7rd34uI3brf7wbcMZ4SJUnDGGaUSwDrgesz850zfvVpYE338xrgU0tfniRpWMPc+n8wcDywISKu6fa9BTgdODciTgS+A7x8PCVKkoYxb6Bn5uVAzPHrI5a2HEnSQnmnqCQ1wkCXpEYY6JLUCANdkhphoEtSIwx0SWqEgS5JjTDQJakRBrokNcJAl6RGGOiS1AgDXZIaYaBLUiMMdElqhIEuSY0w0CWpEQa6JDXCQJekRhjoktQIA12SGmGgS1IjDHRJaoSBLkmNMNAlqREGuiQ1wkCXpEYY6JLUCANdkhphoEtSIwx0SWqEgS5JjTDQJakRBrokNcJAl6RGGOiS1AgDXZIaYaBLUiPmDfSIODMi7oiI62bse2tE/HNEXNN9vXC8ZUqS5jNMC/1DwJGz7P/zzDyg+/rs0pYlSRrVvIGemZcCdy1DLZKkRVhMH/rJEXFt1yWz81wPioi1ETEdEdObNm1axMtJkrZmoYH+XuAXgQOAjcAZcz0wM9dl5urMXD01NbXAl5MkzWdBgZ6Z38vM+zLzfuD9wEFLW5YkaVQLCvSI2G3G5kuB6+Z6rCRpeWw33wMi4mzgMGDXiLgVOBU4LCIOABK4BXj9GGuUJA1h3kDPzONm2b1+DLVIkhbBO0UlqREGuiQ1wkCXpEYY6JLUCANdkhphoEtSIwx0SWrEvOPQJWlUZ7zy6LE+/5s/9g9jff5a2UKXpEYY6JLUCANdkhphoEtSIwx0SWqEgS5JjTDQJakRBrokNcJAl6RGGOiS1AgDXZIaYaBLUiMMdElqhIEuSY0w0CWpEQa6JDXCQJekRhjoktQIA12SGmGgS1IjDHRJaoSBLkmNMNAlqREGuiQ1wkCXpEYY6JLUCANdkhphoEtSI+YN9Ig4MyLuiIjrZuzbJSIujIgbu+87j7dMSdJ8hmmhfwg4cot9pwAXZeYq4KJuW5LUo3kDPTMvBe7aYvcxwFndz2cBL1niuiRJI1poH/rjMnMjQPf9sXM9MCLWRsR0RExv2rRpgS8nSZrP2C+KZua6zFydmaunpqbG/XKStM1aaKB/LyJ2A+i+37F0JUmSFmKhgf5pYE338xrgU0tTjiRpoYYZtng2cAWwV0TcGhEnAqcDz4uIG4HndduSpB5tN98DMvO4OX51xBLXIklaBO8UlaRGGOiS1AgDXZIaYaBLUiMMdElqhIEuSY0w0CWpEQa6JDXCQJekRhjoktQIA12SGmGgS1IjDHRJaoSBLkmNMNAlqREGuiQ1wkCXpEYY6JLUCANdkhphoEtSIwx0SWqEgS5JjTDQJakRBrokNcJAl6RGGOiS1AgDXZIaYaBLUiMMdElqhIEuSY0w0CWpEQa6JDXCQJekRhjoktQIA12SGmGgS1IjDHRJasR2i/njiLgFuBu4D7g3M1cvRVGSpNEtKtA7h2fmnUvwPJKkRbDLRZIasdhAT+CCiLgyItbO9oCIWBsR0xExvWnTpkW+nCRpLosN9IMz8+nArwMnRcShWz4gM9dl5urMXD01NbXIl5MkzWVRgZ6Zt3Xf7wA+CRy0FEVJkka34ECPiEdGxI6Dn4HnA9ctVWGSpNEsZpTL44BPRsTgeT6amZ9fkqokSSNbcKBn5s3A/ktYiyRpERy2KEmNMNAlqREGuiQ1Yilu/Zekptx6ymVjff7dTz9kLM9rC12SGmGgS1IjDHRJaoSBLkmNMNAlqREGuiQ1wkCXpEYY6JLUCANdkhphoEtSIwx0SWqEgS5JjTDQJakRBrokNcJAl6RGGOiS1AgDXZIaYaBLUiMMdElqhIEuSY0w0CWpEQa6JDXCQJekRhjoktQIA12SGmGgS1IjDHRJaoSBLkmNMNAlqREGuiQ1wkCXpEYY6JLUiEUFekQcGRHfiIibIuKUpSpKkjS6BQd6RKwA3gP8OrAvcFxE7LtUhUmSRrOYFvpBwE2ZeXNm3gOcAxyzNGVJkkYVmbmwP4x4GXBkZr6u2z4e+JXMPHmLx60F1nabewHfWHi589oVuHOMzz9u1t+fmmsH6+/buOt/YmZOzfeg7RbxAjHLvp85O2TmOmDdIl5naBExnZmrl+O1xsH6+1Nz7WD9fZuU+hfT5XIr8AsztncHbltcOZKkhVpMoH8VWBURe0bEw4D/AHx6acqSJI1qwV0umXlvRJwMnA+sAM7MzK8vWWULsyxdO2Nk/f2puXaw/r5NRP0LvigqSZos3ikqSY0w0CWpEQa6JDWi6kCPiHMi4gURMduYeGlOEbFL3zVIS63qQAc+BLwW+GZEvC0intxzPSOLiPMi4qiIqPL/RTenT42+HBEfj4gX1togqP2kVPGxM7Hv24kqZlSZ+fnMfCVlXpnbgYsj4tKIOD4iFnMX7HJ6L/Aq4MaIOD0i9u67oBHdFBHvqHBitqdQhpodT/lv+LOIeErPNY2q9pNSrccOTOj7tvphixGxM+Uf9jWUuRQ+CvwqsCozn9tnbaOIiEcBxwF/CHwXeD/wN5n5014Lm0dE7Ei5qewESgPhTOCczPxhr4WNICIOB/4GeCTwNeCUzLyi36rm14X4cymfUg8CPgZ8KDO/2WthQ2rk2Jmo923VgR4R5wL7UUL8g5l564zfXZ2ZB/ZW3Agi4jHAqymtxduAj1BOSvtl5mE9ljaSiDgUOBt4NPC3wH/PzJv6rWp2W/ybfw9YT7nT+QDg45m5Z4/ljazWk9JATcfOwCS+b2vplpjLB4ALc5azUkVh/glgb+DDwIsyc2P3q49FxHR/lQ2n6wc9itLKWgmcQTmwDwE+S+namERXUP7NXzKzIQBMR8T7eqppJLOclN7AjJMSMNEnpYqPnYl931bdQgfo+q72BbYf7MvMj/ZX0Wgi4jmZ+Y9917FQEXEzcDGwPjO/uMXv3pWZv9NPZVsXETFbQ6AmEfFNSqB8cIuTEhHxB5n59n4qG06txw5M7vu26kCPiP8GPJ9ypjwfeAFweWb+Rq+FjSginsrPnpT+ur+KhhcRO2Tmv/Vdx6giYgr4L8Av8eB/9+f0VtSIaj8p1XrsDEzi+7b2LpdXUj5eXpWZx0fEbsBf9VzTSCLiVOAwyoHxWcqSfpcDVQQ6cG9EnMTPBuNr+ytpKB+hXEQ8GvhtYA2wqdeKRrdrRNR8Uqr12JnY923VwxaBH2fmfZQDY0fK0MUn9VzTqF4GHAHcnpknAPsDD++3pJF8GHg85dPRJZR58e/utaLhPCYz1wM/zcxLuhB5Zt9FjegjwA2UvvLTgFso01rXotZjByb0fVt7oF8dEY+mDHeaBr4CXNVvSSP7cWbeTzkp7QTcQV0npSdn5h8BP8rMsygXufbruaZhDIaVbexuEDmQEig1qf2kVOuxAxP6vq26yyUzX9/9+J6IOB/YKTNrC/Tp7qT0fuBK4N8oJ6ZaDILx+12f4u2UEQuT7m3dGOI3A+8GdgLe1G9JI3vQSYkydK6mk1Ktxw5M6Pu2youiEfG0rf0+M69drlqWUkSspJyUqqk/Il4HnAc8DfggsAPwx5lZxdC/mkXE0cBllKUgByel0zKzipXDWjl2Jul9W2ugX9b9+HDgQODrlEWrfwn4amY+q6/ahhURT9/a7yv8pFGFiHg3syxmPjDJQ+XUv0l/31bZ5ZKZhwBExNnA2sy8ptveH3hjn7WN4Izu+/bAasrdfUFprXyZcsfZxIqI39va7zPznctVy4gGN30cTBmh8LFu++WUj84Tr/aTUsXHDkz4+7bKQJ9hn0GYA2Tm1+Y7g06KzDwcyhTAlJPShm77qcB/7rO2Ie3Yfd8LeAabFwh/EXBpLxUNobv4RkT8JnD4YM6N7u7QC3osbRS1n5SqPHZg8t+3VXa5DHRzudxFmcMiKbdBPyYzX9FrYSOIiGsy84D59k2qiLgAODYz7+62d6TMhXJkv5VtXUR8A3hWZt7Vbe8MfCkz9+q3suFFxMXA82eclB4KXDAInUlX67EDk/u+rb2FvgY4GfiDbvtSYKsf5ybQ9RHxAR58Urq+35JGsgdwz4zte6hjpMLplGGvF3fbvwa8tb9yFuQJlNbuXd32Dt2+WtR67ADcMInv26pb6DN1Q4iekJn/p+9aRhER2wP/ETi023Up8N7M/H/9VTW8iPhD4BXAJykH9kuBczPzz3otbAgR8XjgV7rNL2fm7X3WM6qIOIFyEnrQSWnQrTTpKj92JvJ9W3WgR8RFlINgBeXixF2U2Rd/v9fCtjER8ctsvhh0aWZe3Wc9w4iIg4FrMvNHEfFq4OnAX2bmt3subSQNnJRqPHZWAGdl5qv7rmVLtQf61Zl5YEScSPmo9sfA1zJzq+PUJ0FEnJuZr4iIDcwyYqGG/4aB7gB/HDO68DLzO/1VNL+IuJZyu/bTKPNvnAn8Rmb+Wq+FjaCFk1KNxw5AdyPjizLznnkfvIxq70Pfrps17+WUGxIy6lmJazC88uheq1ikiHgDcCplPu77KEO4khKUk+ze7ng5BnhXZq6PiDV9FzWi9wL7d8N1f59yUvprStfLxKv42IEyb84XIuLTwI8GO/secll7oP8pZVKfyzPzKxHxJOBbPdc0lMzc2LVO1mdFS+XN4o3AXpn5L30XMqK7I+K/Ui5mHdr9v3hozzWNqvaTUq3HDpRpFm6jzIe14zyPXTZVB3pmngOcM2P7ZuCY/ioaTWbeFxH/HhGPyswf9F3PAn0XqLH2V1LWoj0xM2+PiD2Ad/Rc06gGJ6XjgUMqPCnVeuyQmaf1XcNsau9DfzLwHuDxmbl/N8fLUZn5P3oubWjdWPpnAhfy4I9uE32330BErKfcIPIZ4CeD/X1/9NyaLvjOr/yT0eCC6Kso011c1p2UDut7kYVh1XjsDHTDXWe79tXrXPRVt9Apa4q+hRLqABsoC81WE+iUg/kzfRexCN/pvh7WfU28Rj4Z0X2yOA9Y1e26kzIEsBbVHTszzLwrdHvgWODenmp5QO0t9K9m5jMGo126fb3frTWMiNijhqv5w+ru8stalhSr/ZMRQET8FrAW2CUzfzEiVgHvy8wjei5tmxQRl/Q9Sqr2Fvq/RMSedB99IuIllDmVa/B3lGFmRMR5mXlsz/UsSDeHxYeBXbrtO4HXZObXey1sfrV/MgI4CTiIMikUmXljRDy235KGN6ndFsOIiF1mbD4E+GXK6ku9qj3QTwbWA3tHxLeBjcBx/ZY0tJnjK3tf6WQR1gG/l5kXA0TEYZRJ/5/dZ1HzqeVuynn8JDPvGQzVjYjt2MosjBNoIrsthnQl5d86KDV/Czix14qoPNAz8ybgOd3KM5GZ3++7phHkHD/X5pGDMAfIzH+KiEf2WdAwIuJbzN46rOnkeklEvAV4REQ8D/hPwN/3XNPQMnPLmSG/EBGX9FLM6PbZ8jb/iOh9TdGqA707mGduA1DDXBCUG0J+SDnDP6L7mW47M3On/kobyc0R8UeUbhco47pruBdg9Yyft6fcnLbLHI+dVKdQWoUbgNdTVp//QK8VjWCWbovVTEC3xZC+SNdlOsMVs+xbVlUHOuXusoHtKYvMTnrfLQCZuaLvGpbIaykrzn+CcjK6FDih14qGMMvNLH8REZdTpo+oQpZFit/ffdVo0G0BpdviFiag22JruqGiP09phB3I5q7TnYCf662wTtWBnplvn7kdEW+nXGzUMsnMfwWqGRkysMVCKIPW4cTc8TeMbi6XtwJPpLyXB5/uJrrbKCKeAXw3M/fsttdQ+s9vASZ9ttQXAL9JWYx75nj5uylDqHtV9bDFLXV96dOZuWreB2tRujks5pSZL16uWhZixjzosLl1+L8y8xv9VDS6iLgBeBOlpfvAp9VJv5U+Iq4CnpuZd0XEoZS7vd8AHEDpm35ZrwUOISKOzczz+q5jS1W20CNiu8y8NyKuZvNHthXAbkAN/ecteBbl1u2zKcPmqpkVDTYvJVa5H2Tm5/ouYgFWDFaKokzBsK4Lx/Mi4pqt/N3EyMzzIuIoysL028/Y/yf9VVVpoANfoVx8mHkmvxe4PTN/MvufaIk9HngeZZjoqyhjus+uYPw58MCIhGMp0y7PnLq11zfkiC6OiHdQrl/MvHW+15Xnh7Bi0CgDjqDcHDVQRSZFWYP254DDKReiX0bJpV5V8Y83iwDIzP/bdyHbqsy8D/g88PkuHI8D/iki/iQz391vdUP5FGViqCuZEYaVGSxsMXPETgKTfmPO2ZQhl3cCPwYugwfmZqplKoZnZ+bTIuLazDwtIs6gnFh7VWugT0XEnGuH1jC5Twu6ID+KEuYrgXcxAQf1kHavYTHiram12ygz/7RbbWw3yqLWg27Th1D60mvw4+77v0fEEyirpe3ZYz1AvYG+grIgblX9ti2JiLOApwKfA07LzOt6LmlUX4yI/TJzQ9+FjGprjRmoo0GTmV+aZd83+6hlgf6hW8f4f1I+5cEE3ANQ5SiXiLgqM3sdwL+ti4j72Typ1cyDaKJvjIqI64D7KY2ZVcDNlC6XQd0Tv1pORJza/bgX8AxgMOLoRZR1OV/XS2HbgBlDLm/vtl9DuZnuBsoC3Xdt7e/HXl+lgf7A7IrSKCLiXynD42ZV2XqcFwDHZubd3faOwMdr70qaZJM+5LLWLhenB9VCfaum0J7HHsDMRYrvoVzL0PhM9JDLKgO97481qtpjG7qg/mHgKxHxSUq310spi0RrfCZ6yGXvBUjLrJkL6t1okc8Bh3S7TsjMq/usaRsw0UMuq+xDlxaqtQvqEfGrwKrM/GBETAE7ZGYNs11WKyKeyeYhlz/q9j2F8m/f601dBrq2KS1dUO9Gu6wG9srMp3TjoT+emQf3XJp68pC+C5CWWUsX1F8KvJhu+Ghm3kZlM0ZqaRno2qY0dkH9nu4uy8GauhO/UpTGy0CX6nVuRPwV8OiI+C3gf1PvYhdaAvahS5WJiN8FvgBcTZnt7/mUUTvnZ+aFfdamfjlsUarP7sBfAnsD11LWt/wCm+cU0TbKFrpUqYh4GGWUy7MpC448C/h+Zu7ba2HqjS10qV6PoCxO/Kju6zagutkjtXRsoUuViYh1lKXP7qYs//cl4Evdgt3ahjnKRarPHsDDgduBfwZuBb7fa0WaCLbQpQpFRFBa6c/uvp5KWTXnisw8dWt/q3YZ6FLFImJ34GBKqB8NPCYzH91vVeqLgS5VJiJ+hxLgBwM/pQxZvKL7viEz7++xPPXIUS5SfVYCfwu8KTM39lyLJogtdElqhKNcJKkRBrokNcJAl6RGGOiS1Ij/DwjwircjlvlRAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6d59a320>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "(test[test['car_type']=='shuttle'][\"travel_day\"].value_counts()/b).plot.bar()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6d61b470>"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6d5e8e80>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "(uber[uber['car_type']=='Bus'][\"travel_day\"].value_counts()/a).plot.bar()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6d643e10>"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6d665358>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "(test[test['car_type']=='Bus'][\"travel_day\"].value_counts()/b).plot.bar()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6d68f5c0>"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6d6f2a90>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "(uber[\"travel_day\"].value_counts()/a).plot.bar()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6d753748>"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6d68f0b8>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "(test[\"travel_day\"].value_counts()/b).plot.bar()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "From the above analysis, there seem to be no clear indication that the day of the week matters as the average people that travel on a particular day seems to change with context."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6d7c01d0>"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6d6bf400>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "uber[uber[\"travel_yr\"]==2017][\"travel_date\"].dt.month.value_counts().sort_index().plot.bar()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6d77eeb8>"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6d7f4630>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "uber[uber[\"travel_yr\"]==2018][\"travel_date\"].dt.month.value_counts().sort_index().plot.bar()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "No data point to plot.\n"
     ]
    }
   ],
   "source": [
    "try:\n",
    "    test[test[\"travel_yr\"]==2017][\"travel_date\"].dt.month.value_counts().sort_index().plot.bar()\n",
    "except:\n",
    "    print('No data point to plot.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6d84afd0>"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6d898eb8>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "test[test[\"travel_yr\"]==2018][\"travel_date\"].dt.month.value_counts().sort_index().plot.bar()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It is clear that from the exploration above that there is something inconsistent about the travel date in the training set. There shouldn't be data for anytime earlier time earlier than Oct 2017 and later than April 2018. This make the date an unreliable indicator. Thus, any date-related feature is unnecessary for modelling.\n",
    "\n",
    "Lastly we can check to see the distribution for ticket sales."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6d4e34e0>"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6d918390>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "((uber[uber['car_type']=='Bus']['number_of_tickets'])).plot.density()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6d9182b0>"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6d90af60>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "((uber[uber['car_type']=='shuttle']['number_of_tickets'])).plot.density()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It can be seen that for buses, the bus is usually almost empty while for the shuttles, they are almost always full or empty."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
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
       "      <th>travel_time</th>\n",
       "      <th>travel_from</th>\n",
       "      <th>car_type</th>\n",
       "      <th>number_of_tickets</th>\n",
       "      <th>hour_booked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>435</td>\n",
       "      <td>Migori</td>\n",
       "      <td>Bus</td>\n",
       "      <td>1.0</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>432</td>\n",
       "      <td>Migori</td>\n",
       "      <td>Bus</td>\n",
       "      <td>1.0</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>425</td>\n",
       "      <td>Keroka</td>\n",
       "      <td>Bus</td>\n",
       "      <td>1.0</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>430</td>\n",
       "      <td>Homa Bay</td>\n",
       "      <td>Bus</td>\n",
       "      <td>5.0</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>432</td>\n",
       "      <td>Migori</td>\n",
       "      <td>Bus</td>\n",
       "      <td>31.0</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   travel_time travel_from car_type  number_of_tickets  hour_booked\n",
       "0          435      Migori      Bus                1.0            7\n",
       "1          432      Migori      Bus                1.0            7\n",
       "2          425      Keroka      Bus                1.0            7\n",
       "3          430    Homa Bay      Bus                5.0            7\n",
       "4          432      Migori      Bus               31.0            7"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "uber = uber[['travel_time','travel_from','car_type','number_of_tickets','hour_booked']]\n",
    "uber.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Trying to linearize the travel time feature for better prediction\n",
    "uber['travel_time_log']=np.log(uber['travel_time'])\n",
    "test['travel_time_log']=np.log(test['travel_time'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We proceed to create two features: late night and early morning based on our EDA."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "uber['early_morning']=uber['hour_booked']<8\n",
    "test['early_morning']=test['hour_booked']<8\n",
    "uber['late_night']=uber['hour_booked']>18\n",
    "test['late_night']=test['hour_booked']>18"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We use the uber .corr function and seaborn's heatmap to see if there is any linear relationships between our features and targets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6d5711d0>"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6d5e8898>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.heatmap(abs(uber.corr()))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There seem to be no strong relationship between all of our features and target.\n",
    "\n",
    "We now try to incoporate an external data - distance from town to Nairobi."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "distance={'Migori':370.9,'Keroka':279.8,'Kisii':305.5,'Homa Bay':305.5,'Keumbu':294.0,\n",
    "        'Rongo':330.3,'Kijauri':276.6,'Oyugis':331.1,'Awendo':349.5,\n",
    "        'Sirare':391.9,'Nyachenge':322.8,'Kehancha':377.5,\n",
    "        'Kendu Bay':367.5,'Sori':392,'Rodi':349.1,'Mbita':399.4,\n",
    "        'Ndhiwa':369.6}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "uber['distance']=uber['travel_from'].map({k:v for k,v in distance.items()})\n",
    "test['distance']=test['travel_from'].map({k:v for k,v in distance.items()})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "test=pd.get_dummies(test,prefix=['car_type','travel_from'],columns=['car_type','travel_from'])\n",
    "uber=pd.get_dummies(uber,prefix=['car_type','travel_from'],columns=['car_type','travel_from'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# MODELLING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original features:\n",
      " ['travel_time', 'number_of_tickets', 'hour_booked', 'travel_time_log', 'early_morning', 'late_night', 'distance', 'car_type_Bus', 'car_type_shuttle', 'travel_from_Awendo', 'travel_from_Homa Bay', 'travel_from_Kehancha', 'travel_from_Kendu Bay', 'travel_from_Keroka', 'travel_from_Keumbu', 'travel_from_Kijauri', 'travel_from_Kisii', 'travel_from_Mbita', 'travel_from_Migori', 'travel_from_Ndhiwa', 'travel_from_Nyachenge', 'travel_from_Oyugis', 'travel_from_Rodi', 'travel_from_Rongo', 'travel_from_Sirare', 'travel_from_Sori'] \n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(\"Original features:\\n\", (list(uber.columns)), \"\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_cols=['travel_time', 'hour_booked', 'travel_time_log', 'early_morning', 'late_night', 'distance', \n",
    "         'car_type_Bus', 'car_type_shuttle', 'travel_from_Awendo', 'travel_from_Homa Bay', 'travel_from_Kehancha',\n",
    "         'travel_from_Kendu Bay', 'travel_from_Keroka', 'travel_from_Keumbu', 'travel_from_Kijauri', \n",
    "         'travel_from_Kisii', 'travel_from_Mbita', 'travel_from_Migori', 'travel_from_Ndhiwa', 'travel_from_Nyachenge',\n",
    "         'travel_from_Oyugis', 'travel_from_Rodi', 'travel_from_Rongo', 'travel_from_Sirare', 'travel_from_Sori']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "predicted_col=['number_of_tickets']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train=uber[feature_cols].values\n",
    "Y_train=uber[predicted_col].values\n",
    "\n",
    "#Reshaping target column to avoid Sklearb throwing in a warning \n",
    "Y_train=Y_train.ravel()\n",
    "\n",
    "split_test_size=0.30\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "Xtrain, Xtest, Ytrain, Ytest= train_test_split(X_train,Y_train, test_size=split_test_size, random_state=260)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import mean_squared_error,mean_absolute_error\n",
    "from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold\n",
    "kfold=KFold(n_splits=5)\n",
    "from sklearn.preprocessing import PolynomialFeatures,MinMaxScaler,StandardScaler\n",
    "poly=PolynomialFeatures(degree=1).fit(Xtrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import GradientBoostingRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.linear_model import LinearRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "Utrain=(poly.transform(Xtrain))\n",
    "Utest=(poly.transform(Xtest))\n",
    "scaler=StandardScaler().fit(Utrain)\n",
    "Utrain=scaler.transform(Utrain)\n",
    "Utest=scaler.transform(Utest)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# First Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " Average CV is:  0.5413819387042605\n",
      "GBR MAE: 3.9239083074393744\n",
      "GBR Training set score: 0.56234\n",
      "GBR Test set score: 0.54731\n"
     ]
    }
   ],
   "source": [
    "gbrt = GradientBoostingRegressor(criterion='mse',random_state=10,n_estimators=100).fit(Utrain,Ytrain)\n",
    "cv = cross_val_score (gbrt,Utrain,Ytrain,cv=5)\n",
    "print(\" Average CV is: \", cv.mean())\n",
    "Ypred=gbrt.predict(Utest)\n",
    "MAE=mean_absolute_error(Ytest,Ypred)\n",
    "MSE=mean_squared_error(Ytest,Ypred)\n",
    "print(\"GBR MAE:\", MAE)\n",
    "print(\"GBR Training set score: {:.5f}\".format(gbrt.score(Utrain,Ytrain)))\n",
    "print(\"GBR Test set score: {:.5f}\".format(gbrt.score(Utest,Ytest)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20d6e2f74e0>"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20d6e2e2b70>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "b=list(gbrt.feature_importances_[1:])\n",
    "pd.DataFrame(index=feature_cols,data=b).plot.bar()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can clearly see that some features are far more important than some others. While we can just minually remove them. It is better we use the sklearn.feature_selection recursive feature selection with or without cross validation tool (it is better with cv)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RFECV(cv=5,\n",
       "   estimator=GradientBoostingRegressor(alpha=0.9, criterion='mse', init=None,\n",
       "             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,\n",
       "             max_leaf_nodes=None, min_impurity_decrease=0.0,\n",
       "             min_impurity_split=None, min_samples_leaf=1,\n",
       "             min_samples_split=...te=10, subsample=1.0, tol=0.0001,\n",
       "             validation_fraction=0.1, verbose=0, warm_start=False),\n",
       "   min_features_to_select=1, n_jobs=None, scoring=None, step=1, verbose=0)"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.feature_selection import SelectFromModel\n",
    "from sklearn.feature_selection import RFE,RFECV\n",
    "select = RFECV(gbrt,cv=5)\n",
    "select.fit(Utrain,Ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "13"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "select.n_features_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We see that the feature selection tool reduced the features from about 25 to 13"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " Average CV is:  0.5434639649815292\n",
      "GBR MAE: 3.9347467695573606\n",
      "GBR Training set score: 0.56809\n",
      "GBR Test set score: 0.54587\n"
     ]
    }
   ],
   "source": [
    "cv = cross_val_score (select,Utrain,Ytrain,cv=5)\n",
    "print(\" Average CV is: \", cv.mean())\n",
    "Ypred=select.predict(Utest)\n",
    "MAE=mean_absolute_error(Ytest,Ypred)\n",
    "print(\"GBR MAE:\", MAE)\n",
    "print(\"GBR Training set score: {:.5f}\".format(select.score(Utrain,Ytrain)))\n",
    "print(\"GBR Test set score: {:.5f}\".format(select.score(Utest,Ytest)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It can be seen that the metrics are almost the same for both sets of features but we prefer the select model because according to Ockham razor principle, you always want to the simplest model that performs best.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Other implementation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "While I tried out other implementations like xgboost, Adaboost, Light GBM, Decision trees, Extra trees and Random Forest, I didnt pay much attention to them as I started the challenge late and didnt have the time to tune every model. I also tried out tricks like PCA but the results were not better off than using just select.I focused only on Gradient Boosting. In hindsight, that may not have been the best decision.\n",
    "\n",
    "I will however be sharing one other implementation I tried out with alongside gradient boosting (after heavy parameter tuning using Grid Search). Because of the time it took to grid-search, I will just be implementing the best model I obtained in my first grid-search range. To learn more about Grid Search ()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RFECV(cv=5,\n",
       "   estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,\n",
       "             learning_rate=0.3, loss='lad', max_depth=3, max_features=None,\n",
       "             max_leaf_nodes=None, min_impurity_decrease=0.0,\n",
       "             min_impurity_split=None, min_samples_leaf=1,\n",
       "             min_samp...=100, subsample=0.75, tol=0.0001,\n",
       "             validation_fraction=0.1, verbose=0, warm_start=False),\n",
       "   min_features_to_select=1, n_jobs=None, scoring=None, step=1, verbose=0)"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "gbr = GradientBoostingRegressor(learning_rate=.3,random_state=100,n_estimators=220,subsample=0.75,\n",
    "                                loss='lad').fit(Utrain,Ytrain)\n",
    "select2 = RFECV(gbr,cv=5)\n",
    "select2.fit(Utrain,Ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " Average CV is:  0.5159643614441931\n",
      "GBR MAE: 3.53976801371126\n",
      "GBR Training set score: 0.54808\n",
      "GBR Test set score: 0.53017\n"
     ]
    }
   ],
   "source": [
    "cv = cross_val_score (select2,Utrain,Ytrain,cv=5)\n",
    "print(\" Average CV is: \", cv.mean())\n",
    "Ypred=select2.predict(Utest)\n",
    "MAE=mean_absolute_error(Ytest,Ypred)\n",
    "print(\"GBR MAE:\", MAE)\n",
    "print(\"GBR Training set score: {:.5f}\".format(select2.score(Utrain,Ytrain)))\n",
    "print(\"GBR Test set score: {:.5f}\".format(select2.score(Utest,Ytest)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "import mlxtend"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [],
   "source": [
    "#You will need to install mlxtend \n",
    "from mlxtend.regressor import StackingCVRegressor"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Since we cant use the stackingCVRegressor with the RFECV select model, we need to redefine our inputs such that the only the most informative features are used. (There is a slight increase from 13 - 19) because I choose to include all the categories of the travel_from. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_cols=['travel_time_log', 'travel_from_Awendo', 'distance','car_type_shuttle',\n",
    "                   'travel_from_Homa Bay', 'travel_from_Kehancha', 'travel_from_Kendu Bay', 'travel_from_Keroka', \n",
    "                   'travel_from_Keumbu', 'travel_from_Kijauri', 'travel_from_Kisii', 'travel_from_Mbita', 'travel_from_Migori', \n",
    "                   'travel_from_Ndhiwa', 'travel_from_Nyachenge', 'travel_from_Rodi', \n",
    "                   'travel_from_Rongo', 'travel_from_Sirare', 'travel_from_Sori']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "predicted_col=['number_of_tickets']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train=uber[feature_cols].values\n",
    "Y_train=uber[predicted_col].values\n",
    "\n",
    "Y_train=Y_train.ravel()\n",
    "\n",
    "split_test_size=0.3\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "Xtrain, Xtest, Ytrain, Ytest= train_test_split(X_train,Y_train, test_size=split_test_size, random_state=260)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Stacking is a type of ensembling that combines the results of two or more estimators using another estimator. Please note that my implementation may not be the best. Stacking is supposed to be used when you are trying to merge the results of three very good estimators. I didn't optimize the decision tree and random forest models.\n",
    "\n",
    "However, it can be seen that stacked model is not so far off from my best model (imagine the potential if I had used it on many highly tuned models)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "StackingCVRegressor(cv=5,\n",
       "          meta_regressor=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,\n",
       "         normalize=False),\n",
       "          refit=True,\n",
       "          regressors=(GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,\n",
       "             learning_rate=0.3, loss='lad', max_depth=3, max_features=None,\n",
       "             max_leaf_nodes=None, min_impurity_decrease=0.0,\n",
       "             min_impurity_split=None, min_samples_leaf=1,\n",
       "             min_sa...imators=100, n_jobs=None,\n",
       "           oob_score=False, random_state=10, verbose=0, warm_start=False)),\n",
       "          shuffle=True, store_train_meta_features=False,\n",
       "          use_features_in_secondary=False)"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lr=LinearRegression()\n",
    "dt = DecisionTreeRegressor(criterion='mae',random_state=100)\n",
    "rf = RandomForestRegressor(random_state=10,n_estimators=100)\n",
    "gb = GradientBoostingRegressor(learning_rate=.3,random_state=100,n_estimators=220,subsample=0.75,\n",
    "                               loss='lad')\n",
    "stack = StackingCVRegressor(regressors=(gb, dt, rf),\n",
    "                            meta_regressor=lr,cv=5)\n",
    "stack.fit(Utrain,Ytrain)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Below I will be slightly changing my implementation. I am using the mean_absolute_error as the scorer so I can easily see whether my model is generalizing well since mae is the objective metric is the challenge."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import make_scorer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average CV is: 3.757 0.20927656236807862\n",
      "GBR Training set score: 3.604\n",
      "GBR Test set score: 3.742\n"
     ]
    }
   ],
   "source": [
    "cv = cross_val_score (stack,Utrain,Ytrain,cv=5,scoring=make_scorer(mean_absolute_error))\n",
    "print(\"Average CV is:\", round(cv.mean(),3),cv.std())\n",
    "Ypred=stack.predict(Utest)\n",
    "Ypred_t=stack.predict(Utrain)\n",
    "MAE=mean_absolute_error(Ytest,Ypred)\n",
    "MAE_t=mean_absolute_error(Ytrain,Ypred_t)\n",
    "print(\"GBR Training set score: {:.3f}\".format(MAE_t))\n",
    "print(\"GBR Test set score: {:.3f}\".format(MAE))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "After another set of gridsearch (ran for about five hours), my winning I came up with my best solution which ended out in the top 25% of all the submitted entries. My model ended up about 0.5 MAE behind winning model in the public leaderboard. I ran out of time to try out other grid search parameters unfortunately.\n",
    "\n",
    "Below I will contrasting my model to a model of a friend that ended in the top 5% of all submitted entries (about 0.2 MAE behind the winning model)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average CV is: 3.617 0.2584669864383674\n",
      "GBR Training set score: 3.425\n",
      "GBR Test set score: 3.512\n"
     ]
    }
   ],
   "source": [
    "gb=GradientBoostingRegressor(learning_rate=.5,random_state=100,n_estimators=250,subsample=0.75,loss='lad',\n",
    "                            max_depth=4).fit(Xtrain,Ytrain)\n",
    "cv = cross_val_score (gb,Utrain,Ytrain,cv=5,scoring=make_scorer(mean_absolute_error))\n",
    "print(\"Average CV is:\", round(cv.mean(),3),cv.std())\n",
    "Ypred=gb.predict(Xtest)\n",
    "Ypred_t=gb.predict(Xtrain)\n",
    "MAE=mean_absolute_error(Ytest,Ypred)\n",
    "MAE_t=mean_absolute_error(Ytrain,Ypred_t)\n",
    "print(\"GBR Training set score: {:.3f}\".format(MAE_t))\n",
    "print(\"GBR Test set score: {:.3f}\".format(MAE))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The major difference between the two implementation is the range of Grid Search. While I constrained myself, his grid search was more extensive but it took about 3 days for these parameters to be obtained. His implementation is available in the folder. \n",
    "\n",
    "Finally, I will be sharing something I learnt from a friend after I shared this concern with him after the competition ended: Randomized Search. Randomized Search is similar to Grid Search. The only difference is not available permutations are tested. It randomly picks a specified amount of permutations as defined by you. This could save you a lot of time and helps you choose a more extensive range of search parameters. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import RandomizedSearchCV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ADEBAYO\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_search.py:791: RuntimeWarning: overflow encountered in square\n",
      "  array_means[:, np.newaxis]) ** 2,\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "RandomizedSearchCV(cv=3, error_score='raise-deprecating',\n",
       "          estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,\n",
       "             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,\n",
       "             max_leaf_nodes=None, min_impurity_decrease=0.0,\n",
       "             min_impurity_split=None, min_samples_leaf=1,\n",
       "             min_sampl...te=12, subsample=1.0, tol=0.0001,\n",
       "             validation_fraction=0.1, verbose=0, warm_start=False),\n",
       "          fit_params=None, iid='warn', n_iter=10, n_jobs=-1,\n",
       "          param_distributions={'learning_rate': [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 5], 'n_estimators': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 43...', 'ls', 'huber'], 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]},\n",
       "          pre_dispatch='2*n_jobs', random_state=81, refit=True,\n",
       "          return_train_score=True, scoring=None, verbose=0)"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#There are about 300,000 different combinations in the grid defined below. We would be using random search to pick just ten\n",
    "#and see how our model fairs (we do tgis using n_iter)\n",
    "\n",
    "estimator = GradientBoostingRegressor(random_state=12)\n",
    "param = {'learning_rate':[0.001, 0.003,.01,0.03,0.05,0.1,0.3,0.5,1,3,5 ],\n",
    "        'n_estimators':[i for i in range(50,550,10)],\n",
    "        'subsample':[i/100 for i in range(50,100,5)],\n",
    "        'loss':['lad','ls','huber'],\n",
    "        'max_depth':[i for i in range(1,20)]}\n",
    "rs=RandomizedSearchCV(estimator, param_distributions = param, \n",
    "                      n_iter=10, n_jobs=-1, random_state=81,cv=3,\n",
    "                     return_train_score=True)\n",
    "rs.fit(Xtrain,Ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
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
       "      <th>8</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>4</th>\n",
       "      <th>9</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>mean_fit_time</th>\n",
       "      <td>9.46327</td>\n",
       "      <td>18.5958</td>\n",
       "      <td>4.23771</td>\n",
       "      <td>20.7244</td>\n",
       "      <td>0.421857</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std_fit_time</th>\n",
       "      <td>0.454136</td>\n",
       "      <td>0.103495</td>\n",
       "      <td>0.241539</td>\n",
       "      <td>0.899712</td>\n",
       "      <td>0.0127586</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean_score_time</th>\n",
       "      <td>0.046871</td>\n",
       "      <td>0.0520789</td>\n",
       "      <td>0.0260406</td>\n",
       "      <td>0.0572892</td>\n",
       "      <td>0.00520714</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std_score_time</th>\n",
       "      <td>0.022094</td>\n",
       "      <td>0.0073653</td>\n",
       "      <td>0.00736547</td>\n",
       "      <td>0.00736648</td>\n",
       "      <td>0.00736401</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>param_subsample</th>\n",
       "      <td>0.75</td>\n",
       "      <td>0.5</td>\n",
       "      <td>0.6</td>\n",
       "      <td>0.65</td>\n",
       "      <td>0.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>param_n_estimators</th>\n",
       "      <td>330</td>\n",
       "      <td>230</td>\n",
       "      <td>350</td>\n",
       "      <td>450</td>\n",
       "      <td>100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>param_max_depth</th>\n",
       "      <td>7</td>\n",
       "      <td>18</td>\n",
       "      <td>4</td>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>param_loss</th>\n",
       "      <td>lad</td>\n",
       "      <td>lad</td>\n",
       "      <td>lad</td>\n",
       "      <td>lad</td>\n",
       "      <td>lad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>param_learning_rate</th>\n",
       "      <td>0.01</td>\n",
       "      <td>0.05</td>\n",
       "      <td>0.1</td>\n",
       "      <td>0.1</td>\n",
       "      <td>0.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>params</th>\n",
       "      <td>{'subsample': 0.75, 'n_estimators': 330, 'max_...</td>\n",
       "      <td>{'subsample': 0.5, 'n_estimators': 230, 'max_d...</td>\n",
       "      <td>{'subsample': 0.6, 'n_estimators': 350, 'max_d...</td>\n",
       "      <td>{'subsample': 0.65, 'n_estimators': 450, 'max_...</td>\n",
       "      <td>{'subsample': 0.5, 'n_estimators': 100, 'max_d...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>split0_test_score</th>\n",
       "      <td>0.582917</td>\n",
       "      <td>0.584482</td>\n",
       "      <td>0.583257</td>\n",
       "      <td>0.581858</td>\n",
       "      <td>0.383814</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>split1_test_score</th>\n",
       "      <td>0.518422</td>\n",
       "      <td>0.518844</td>\n",
       "      <td>0.51689</td>\n",
       "      <td>0.516053</td>\n",
       "      <td>0.358329</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>split2_test_score</th>\n",
       "      <td>0.481112</td>\n",
       "      <td>0.468336</td>\n",
       "      <td>0.463358</td>\n",
       "      <td>0.457859</td>\n",
       "      <td>0.330896</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean_test_score</th>\n",
       "      <td>0.527484</td>\n",
       "      <td>0.523888</td>\n",
       "      <td>0.521168</td>\n",
       "      <td>0.51859</td>\n",
       "      <td>0.35768</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std_test_score</th>\n",
       "      <td>0.042053</td>\n",
       "      <td>0.0475503</td>\n",
       "      <td>0.0490421</td>\n",
       "      <td>0.0506545</td>\n",
       "      <td>0.0216084</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>rank_test_score</th>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>split0_train_score</th>\n",
       "      <td>0.532657</td>\n",
       "      <td>0.548755</td>\n",
       "      <td>0.532041</td>\n",
       "      <td>0.546197</td>\n",
       "      <td>0.36262</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>split1_train_score</th>\n",
       "      <td>0.562201</td>\n",
       "      <td>0.576943</td>\n",
       "      <td>0.566888</td>\n",
       "      <td>0.572449</td>\n",
       "      <td>0.382529</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>split2_train_score</th>\n",
       "      <td>0.571901</td>\n",
       "      <td>0.586415</td>\n",
       "      <td>0.569794</td>\n",
       "      <td>0.579995</td>\n",
       "      <td>0.386988</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean_train_score</th>\n",
       "      <td>0.555587</td>\n",
       "      <td>0.570704</td>\n",
       "      <td>0.556241</td>\n",
       "      <td>0.566214</td>\n",
       "      <td>0.377379</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std_train_score</th>\n",
       "      <td>0.0166902</td>\n",
       "      <td>0.0159952</td>\n",
       "      <td>0.017153</td>\n",
       "      <td>0.0144853</td>\n",
       "      <td>0.0105941</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                                     8  \\\n",
       "mean_fit_time                                                  9.46327   \n",
       "std_fit_time                                                  0.454136   \n",
       "mean_score_time                                               0.046871   \n",
       "std_score_time                                                0.022094   \n",
       "param_subsample                                                   0.75   \n",
       "param_n_estimators                                                 330   \n",
       "param_max_depth                                                      7   \n",
       "param_loss                                                         lad   \n",
       "param_learning_rate                                               0.01   \n",
       "params               {'subsample': 0.75, 'n_estimators': 330, 'max_...   \n",
       "split0_test_score                                             0.582917   \n",
       "split1_test_score                                             0.518422   \n",
       "split2_test_score                                             0.481112   \n",
       "mean_test_score                                               0.527484   \n",
       "std_test_score                                                0.042053   \n",
       "rank_test_score                                                      1   \n",
       "split0_train_score                                            0.532657   \n",
       "split1_train_score                                            0.562201   \n",
       "split2_train_score                                            0.571901   \n",
       "mean_train_score                                              0.555587   \n",
       "std_train_score                                              0.0166902   \n",
       "\n",
       "                                                                     1  \\\n",
       "mean_fit_time                                                  18.5958   \n",
       "std_fit_time                                                  0.103495   \n",
       "mean_score_time                                              0.0520789   \n",
       "std_score_time                                               0.0073653   \n",
       "param_subsample                                                    0.5   \n",
       "param_n_estimators                                                 230   \n",
       "param_max_depth                                                     18   \n",
       "param_loss                                                         lad   \n",
       "param_learning_rate                                               0.05   \n",
       "params               {'subsample': 0.5, 'n_estimators': 230, 'max_d...   \n",
       "split0_test_score                                             0.584482   \n",
       "split1_test_score                                             0.518844   \n",
       "split2_test_score                                             0.468336   \n",
       "mean_test_score                                               0.523888   \n",
       "std_test_score                                               0.0475503   \n",
       "rank_test_score                                                      2   \n",
       "split0_train_score                                            0.548755   \n",
       "split1_train_score                                            0.576943   \n",
       "split2_train_score                                            0.586415   \n",
       "mean_train_score                                              0.570704   \n",
       "std_train_score                                              0.0159952   \n",
       "\n",
       "                                                                     2  \\\n",
       "mean_fit_time                                                  4.23771   \n",
       "std_fit_time                                                  0.241539   \n",
       "mean_score_time                                              0.0260406   \n",
       "std_score_time                                              0.00736547   \n",
       "param_subsample                                                    0.6   \n",
       "param_n_estimators                                                 350   \n",
       "param_max_depth                                                      4   \n",
       "param_loss                                                         lad   \n",
       "param_learning_rate                                                0.1   \n",
       "params               {'subsample': 0.6, 'n_estimators': 350, 'max_d...   \n",
       "split0_test_score                                             0.583257   \n",
       "split1_test_score                                              0.51689   \n",
       "split2_test_score                                             0.463358   \n",
       "mean_test_score                                               0.521168   \n",
       "std_test_score                                               0.0490421   \n",
       "rank_test_score                                                      3   \n",
       "split0_train_score                                            0.532041   \n",
       "split1_train_score                                            0.566888   \n",
       "split2_train_score                                            0.569794   \n",
       "mean_train_score                                              0.556241   \n",
       "std_train_score                                               0.017153   \n",
       "\n",
       "                                                                     4  \\\n",
       "mean_fit_time                                                  20.7244   \n",
       "std_fit_time                                                  0.899712   \n",
       "mean_score_time                                              0.0572892   \n",
       "std_score_time                                              0.00736648   \n",
       "param_subsample                                                   0.65   \n",
       "param_n_estimators                                                 450   \n",
       "param_max_depth                                                     10   \n",
       "param_loss                                                         lad   \n",
       "param_learning_rate                                                0.1   \n",
       "params               {'subsample': 0.65, 'n_estimators': 450, 'max_...   \n",
       "split0_test_score                                             0.581858   \n",
       "split1_test_score                                             0.516053   \n",
       "split2_test_score                                             0.457859   \n",
       "mean_test_score                                                0.51859   \n",
       "std_test_score                                               0.0506545   \n",
       "rank_test_score                                                      4   \n",
       "split0_train_score                                            0.546197   \n",
       "split1_train_score                                            0.572449   \n",
       "split2_train_score                                            0.579995   \n",
       "mean_train_score                                              0.566214   \n",
       "std_train_score                                              0.0144853   \n",
       "\n",
       "                                                                     9  \n",
       "mean_fit_time                                                 0.421857  \n",
       "std_fit_time                                                 0.0127586  \n",
       "mean_score_time                                             0.00520714  \n",
       "std_score_time                                              0.00736401  \n",
       "param_subsample                                                    0.5  \n",
       "param_n_estimators                                                 100  \n",
       "param_max_depth                                                      1  \n",
       "param_loss                                                         lad  \n",
       "param_learning_rate                                                0.3  \n",
       "params               {'subsample': 0.5, 'n_estimators': 100, 'max_d...  \n",
       "split0_test_score                                             0.383814  \n",
       "split1_test_score                                             0.358329  \n",
       "split2_test_score                                             0.330896  \n",
       "mean_test_score                                                0.35768  \n",
       "std_test_score                                               0.0216084  \n",
       "rank_test_score                                                      5  \n",
       "split0_train_score                                             0.36262  \n",
       "split1_train_score                                            0.382529  \n",
       "split2_train_score                                            0.386988  \n",
       "mean_train_score                                              0.377379  \n",
       "std_train_score                                              0.0105941  "
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a=pd.DataFrame(rs.cv_results_)\n",
    "a.sort_values('rank_test_score').head().transpose()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ADEBAYO\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.\n",
      "  DeprecationWarning)\n",
      "C:\\Users\\ADEBAYO\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_search.py:791: RuntimeWarning: overflow encountered in square\n",
      "  array_means[:, np.newaxis]) ** 2,\n",
      "C:\\Users\\ADEBAYO\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.\n",
      "  DeprecationWarning)\n",
      "C:\\Users\\ADEBAYO\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_search.py:791: RuntimeWarning: overflow encountered in square\n",
      "  array_means[:, np.newaxis]) ** 2,\n"
     ]
    }
   ],
   "source": [
    "cv = cross_val_score (rs,Utrain,Ytrain,cv=5,scoring=make_scorer(mean_absolute_error))\n",
    "print(\"Average CV is:\", round(cv.mean(),3),cv.std())\n",
    "Ypred=rs.predict(Xtest)\n",
    "Ypred_t=rs.predict(Xtrain)\n",
    "MAE=mean_absolute_error(Ytest,Ypred)\n",
    "MAE_t=mean_absolute_error(Ytrain,Ypred_t)\n",
    "print(\"GBR Training set score: {:.3f}\".format(MAE_t))\n",
    "print(\"GBR Test set score: {:.3f}\".format(MAE))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It can be seen that while the implementation above is somewhat simple and time-efficient. The metrics are not particularly bad. One can rerun the search a few number of times and see whether there is a trend in the chosen parameter and then grid search based on the smaller range. One could also random search for a larger number of times."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Conclusion"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "I heard that a top 5 model used Genetic Algorithm for its parameter tuning. Apparently, the person had a lot of time for the tuning as GA is very time-consuming like Grid Search. This may not be possible in a short hackathons or even in some real life scenerios where time is severly limited.\n",
    "\n",
    "This notebook have taken us through some very important concepts in data science (EDA, Feature Selection, Stacking, HypperParameter Tuning) using the uber data from Zindi. It is specifically desined for those who are just starting their data science journey especially as regards working with real-life data. I hope you find it informative.\n",
    "\n",
    "P.S: Any other top solution could just clone this notebook and mention one or two two things that made their model stand out if they dont have the time to share an extensive notebook like this. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
   "version": "3.6.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

